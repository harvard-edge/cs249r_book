"""
``binder layout`` — PDF page-layout diagnostics.

Subcommands:
    check    — Scan a built PDF for pages with excessive bottom whitespace
               in the main body column, and guess the likely cause (the
               block at the top of the next page that probably forced the
               break).

The check is read-only: it inspects the PDF, prints a report, and exits.
It does not modify source or PDF. Fixes happen in the QMD source by the
author, optionally guided by a /layout-fix skill.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()


# --- geometry constants ---------------------------------------------------
# These are heuristic bands for the book's Tufte-style layout (main column
# left, margin notes right). They are intentionally generous so the tool
# works without per-volume calibration; tune later if needed.

HEADER_FRAC = 0.06   # top 6% of page height treated as running header
FOOTER_FRAC = 0.06   # bottom 6% treated as running footer / page number
MAIN_COL_RIGHT_FRAC = 0.55  # main column ends ~55% across the page
LEFT_MARGIN_RIGHT_FRAC = 0.22
RIGHT_MARGIN_LEFT_FRAC = 0.78
# Margin material alternates sides in the bound PDF: verso pages use the left
# band and recto pages use the right band.

PDF_BODY_WIDTH_PT = 8.0 * 72.0 - (0.875 + 1.75) * 72.0
TABLE_MARKER_RE = re.compile(r"\b(?P<idx>\d{3})\s+(?P<label>tbl-[A-Za-z0-9_-]+)\b")


@dataclass
class PageReport:
    sheet: int                    # 1-indexed PDF sheet number
    label: str                    # printed page number ("59", "iii", etc.)
    chapter: str                  # enclosing top-level outline title
    whitespace_pts: float         # vertical gap, in PDF points
    whitespace_frac: float        # gap / usable_text_height
    body_bottom_y: float          # y of last char in main column
    usable_bottom_y: float        # y of footer band top
    cause: str                    # best-effort guess for next-page culprit
    detail: str = ""              # caption / heading / title of the next-page element
    size_hint: str = ""           # element size: "10 rows", "380pt tall", etc.
    fix_hint: str = ""            # one-word suggested fix
    is_frontmatter: bool = False  # roman-numbered page (likely intentional gap)
    # ---- LLM-actionable fields ----
    source_file: str = ""         # QMD chapter file, relative to repo root
    source_line: int = 0          # 1-indexed line where the detail text appears
    section: str = ""             # nearest preceding H2/H3 header text
    next_page_starts_chapter: bool = False   # next sheet is a chapter opener
    klass: str = ""               # A | B | C | D classification
    action: str = ""              # recommended action (try-move-up / accept-…)


@dataclass
class MarginFinding:
    """A margin figure/note layout finding."""
    sheet: int                    # 1-indexed PDF sheet number
    label: str                    # printed page number
    chapter: str                  # enclosing chapter title
    signal: str                   # "image", "caption/text", or both
    over_pts: float               # how far past the footer line (pts); + = below
    off_page: bool                # content extends beyond the physical page edge
    snippet: str = ""             # lowest margin-line text (for grepping source)
    source_file: str = ""         # QMD chapter file, relative to repo root
    source_line: int = 0          # 1-indexed line of the matching caption
    section: str = ""             # nearest preceding H2/H3 header
    issue: str = "overflow"       # overflow | baseline-crowding | image-text-overlap
    severity: str = "error"       # error blocks the margin gate; warning is audit-only
    related: str = ""             # paired text/object snippet for overlap findings


@dataclass
class CollisionFinding:
    """A header/footer collision finding."""
    sheet: int                    # 1-indexed PDF sheet number
    label: str                    # printed page number
    chapter: str                  # enclosing chapter title
    band: str                     # "header" or "footer"
    y: float                      # top coordinate of the colliding text line
    snippet: str                  # colliding line text


@dataclass
class TableAuditEntry:
    """One source table included in a table-only PDF audit."""
    index: int
    label: str
    caption: str
    source_file: str
    source_line: int
    qmd_path: Path
    table_markdown: str
    caption_markdown: str
    columns: int
    rows: int
    max_cell_chars: int
    has_colwidths: bool
    colwidths: str = ""
    rendered_sheet: int = 0
    rendered_width_pt: float = 0.0
    rendered_width_frac: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "caption": self.caption,
            "columns": self.columns,
            "rows": self.rows,
            "max_cell_chars": self.max_cell_chars,
            "has_colwidths": self.has_colwidths,
            "colwidths": self.colwidths,
            "rendered_sheet": self.rendered_sheet,
            "rendered_width_pt": round(self.rendered_width_pt, 1),
            "rendered_width_frac": round(self.rendered_width_frac, 3),
            "warnings": self.warnings,
        }


class LayoutCommand:
    """Diagnose PDF page-break whitespace issues."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    # ------------------------------------------------------------------
    # entry
    # ------------------------------------------------------------------

    def run(self, args: List[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder layout",
            description="PDF page-layout diagnostics.",
            add_help=True,
        )
        sub = parser.add_subparsers(dest="subcommand")

        collisions = sub.add_parser(
            "collisions",
            help="Scan PDF for body content invading header / footer bands.",
        )
        collisions.add_argument("pdf", help="Path to PDF file to scan.")
        collisions.add_argument(
            "--chapter",
            type=str,
            default="",
            help="Only scan outline chapters matching this comma-separated "
                 "title/slug/substring filter.",
        )
        collisions.add_argument(
            "--csv",
            action="store_true",
            help="Emit one CSV row per collision: chapter,sheet,label,band,y,"
                 "snippet.",
        )

        check = sub.add_parser("check", help="Scan PDF for whitespace gaps.")
        check.add_argument("pdf", help="Path to PDF file to scan.")
        check.add_argument(
            "--chapter",
            type=str,
            default="",
            help="Only scan outline chapters matching this comma-separated "
                 "title/slug/substring filter.",
        )
        check.add_argument(
            "--threshold",
            type=float,
            default=0.25,
            help="Flag pages where bottom whitespace exceeds this "
                 "fraction of usable text height (default 0.25).",
        )
        check.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Only scan the first N pages (0 = all). For quick iteration.",
        )
        check.add_argument(
            "--only",
            type=str,
            default="",
            help="Comma-separated culprit classes to keep "
                 "(table,figure,heading,paragraph,callout/box,unknown,"
                 "end-of-document). Default: all.",
        )
        check.add_argument(
            "--skip-frontmatter",
            action="store_true",
            help="Hide pages with roman-numeral labels (frontmatter / "
                 "intentional whitespace).",
        )
        check.add_argument(
            "--csv",
            action="store_true",
            help="Emit one CSV row per flagged page to stdout instead of "
                 "the rich report. Columns: chapter,sheet,label,gap_pct,"
                 "pts,culprit,detail,size_hint,fix_hint,is_frontmatter.",
        )

        margins = sub.add_parser(
            "margins",
            help="Scan PDF for margin figures/notes overflowing off the page "
                 "(exits non-zero when any overflow is found).",
        )
        margins.add_argument("pdf", help="Path to PDF file to scan.")
        margins.add_argument(
            "--chapter",
            type=str,
            default="",
            help="Only scan outline chapters matching this comma-separated "
                 "title/slug/substring filter.",
        )
        margins.add_argument(
            "--tol",
            type=float,
            default=2.0,
            help="Points of slack below the footer line before margin "
                 "content counts as overflow (default 2.0).",
        )
        margins.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Only scan the first N pages (0 = all). For quick iteration.",
        )
        margins.add_argument(
            "--include-overlaps",
            action="store_true",
            help="Also emit warning-level margin text crowding and image/text "
                 "overlap candidates. Footer/off-page overflow remains the "
                 "blocking gate.",
        )
        margins.add_argument(
            "--csv",
            action="store_true",
            help="Emit one CSV row per finding: chapter,sheet,label,issue,"
                 "severity,signal,over_pts,off_page,source_file,source_line,"
                 "section,snippet,related.",
        )

        tables = sub.add_parser(
            "tables",
            help="Render a table-only PDF audit for a volume or chapter.",
        )
        vol_group = tables.add_mutually_exclusive_group(required=True)
        vol_group.add_argument(
            "--vol1",
            dest="volume",
            action="store_const",
            const="vol1",
            help="Audit Volume I tables.",
        )
        vol_group.add_argument(
            "--vol2",
            dest="volume",
            action="store_const",
            const="vol2",
            help="Audit Volume II tables.",
        )
        tables.add_argument(
            "--chapter",
            type=str,
            default="",
            help="Only include these comma-separated chapter stems/slugs.",
        )
        tables.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Only include the first N extracted tables (0 = all).",
        )
        tables.add_argument(
            "--min-width",
            type=float,
            default=0.68,
            help="Advisory warning threshold for rendered table width as a "
                 "fraction of the main text block (default 0.68).",
        )
        tables.add_argument(
            "--fail-under",
            type=float,
            default=0.0,
            help="Exit non-zero when any measured table renders below this "
                 "width fraction. Default 0 keeps the audit advisory.",
        )
        tables.add_argument(
            "--no-render",
            action="store_true",
            help="Extract tables and write tables.qmd/json/csv without "
                 "running Quarto.",
        )
        tables.add_argument(
            "--no-contact-sheets",
            action="store_true",
            help="Skip PNG contact sheet generation after rendering.",
        )
        tables.add_argument(
            "--contact-cols",
            type=int,
            default=3,
            help="Contact-sheet thumbnail columns (default 3).",
        )
        tables.add_argument(
            "--contact-rows",
            type=int,
            default=4,
            help="Contact-sheet thumbnail rows (default 4).",
        )
        tables.add_argument(
            "--dpi",
            type=int,
            default=110,
            help="PDF rasterization DPI for contact sheets (default 110).",
        )

        if not args:
            parser.print_help()
            return False

        opts = parser.parse_args(args)
        if opts.subcommand == "collisions":
            return self._collisions(
                Path(opts.pdf),
                chapter_filter=self._parse_chapter_filter(opts.chapter),
                csv=opts.csv,
            )
        if opts.subcommand == "margins":
            return self._margins(
                Path(opts.pdf),
                tol=opts.tol,
                limit=opts.limit,
                csv=opts.csv,
                chapter_filter=self._parse_chapter_filter(opts.chapter),
                include_overlaps=opts.include_overlaps,
            )
        if opts.subcommand == "check":
            only = (
                set(s.strip() for s in opts.only.split(",") if s.strip())
                if opts.only
                else set()
            )
            return self._check(
                Path(opts.pdf),
                opts.threshold,
                opts.limit,
                only=only,
                skip_frontmatter=opts.skip_frontmatter,
                csv=opts.csv,
                chapter_filter=self._parse_chapter_filter(opts.chapter),
            )
        if opts.subcommand == "tables":
            return self._tables(
                opts.volume,
                chapter_filter=self._parse_chapter_filter(opts.chapter),
                limit=opts.limit,
                min_width=opts.min_width,
                fail_under=opts.fail_under,
                render=not opts.no_render,
                contact_sheets=not opts.no_contact_sheets,
                contact_cols=max(1, opts.contact_cols),
                contact_rows=max(1, opts.contact_rows),
                dpi=max(36, opts.dpi),
            )

        parser.print_help()
        return False

    # ------------------------------------------------------------------
    # tables
    # ------------------------------------------------------------------

    def _tables(
        self,
        volume: str,
        chapter_filter: Optional[List[str]] = None,
        limit: int = 0,
        min_width: float = 0.68,
        fail_under: float = 0.0,
        render: bool = True,
        contact_sheets: bool = True,
        contact_cols: int = 3,
        contact_rows: int = 4,
        dpi: int = 110,
    ) -> bool:
        """Build a table-only PDF audit using the production PDF geometry."""
        repo_root = self._repo_root()
        out_dir = repo_root / "book" / ".layout" / "tables" / volume
        out_dir.mkdir(parents=True, exist_ok=True)

        qmd_files = self._table_source_files(volume, chapter_filter)
        if not qmd_files:
            console.print(
                f"[red]No QMD files found for {volume}"
                f"{' / ' + ','.join(chapter_filter or []) if chapter_filter else ''}.[/red]"
            )
            return False

        entries: List[TableAuditEntry] = []
        for qmd in qmd_files:
            entries.extend(self._extract_pipe_tables(qmd))

        if limit > 0:
            entries = entries[:limit]
        for i, entry in enumerate(entries, start=1):
            entry.index = i

        if not entries:
            console.print("[yellow]No captioned pipe tables found.[/yellow]")
            return False

        qmd_path = out_dir / "tables.qmd"
        pdf_path = qmd_path.with_suffix(".pdf")
        json_path = out_dir / "tables.json"
        csv_path = out_dir / "tables.csv"
        render_log = out_dir / "quarto-render.log"

        self._write_table_audit_qmd(qmd_path, entries, volume)
        console.print(
            f"[bold blue]Extracted[/bold blue] {len(entries)} table"
            f"{'s' if len(entries) != 1 else ''} from {len(qmd_files)} file"
            f"{'s' if len(qmd_files) != 1 else ''}."
        )
        console.print(f"[dim]Audit source:[/dim] {qmd_path}")

        render_ok = True
        overfull_count = 0
        if render:
            render_ok = self._render_table_audit_pdf(qmd_path, render_log)
            if not render_ok:
                console.print(f"[red]Render failed.[/red] See {render_log}")
                return False
            overfull_count = self._count_overfull_hboxes(qmd_path.with_suffix(".log"))
            if pdf_path.exists():
                self._measure_table_pdf(pdf_path, entries, min_width=min_width)
                if contact_sheets:
                    sheets = self._make_table_contact_sheets(
                        pdf_path,
                        out_dir / "contact-sheets",
                        cols=contact_cols,
                        rows=contact_rows,
                        dpi=dpi,
                    )
                    if sheets:
                        console.print("[dim]Contact sheets:[/dim]")
                        for sheet in sheets:
                            console.print(f"  {sheet}")
            else:
                console.print(f"[red]Expected PDF not found:[/red] {pdf_path}")
                return False
        else:
            console.print("[yellow]Skipped render (--no-render).[/yellow]")

        self._write_table_reports(entries, json_path, csv_path)
        self._render_table_summary(
            entries,
            pdf_path if pdf_path.exists() else None,
            json_path,
            csv_path,
            min_width=min_width,
            overfull_count=overfull_count,
            rendered=render,
        )

        if fail_under > 0:
            offenders = [
                e for e in entries
                if e.rendered_width_frac and e.rendered_width_frac < fail_under
            ]
            if offenders:
                console.print(
                    f"[red]✗ {len(offenders)} table"
                    f"{'s' if len(offenders) != 1 else ''} rendered below "
                    f"--fail-under {fail_under:.2f}[/red]"
                )
                return False
        return render_ok

    def _repo_root(self) -> Path:
        root = Path(self.config_manager.root_dir).resolve()
        if (root / "book" / "quarto").is_dir():
            return root
        if (root / "quarto").is_dir():
            return root.parent
        return root

    def _table_source_files(
        self,
        volume: str,
        chapter_filter: Optional[List[str]] = None,
    ) -> List[Path]:
        if chapter_filter:
            out = []
            for raw in chapter_filter:
                spec = raw if raw.startswith(("vol1/", "vol2/")) else f"{volume}/{raw}"
                qmd = self.chapter_discovery.find_chapter_file(
                    spec,
                    allow_fuzzy=True,
                )
                if qmd is None:
                    console.print(f"[yellow]Could not resolve chapter:[/yellow] {raw}")
                    continue
                out.append(qmd)
            return self._dedupe_paths(out)

        config_path = (
            self.config_manager.book_dir
            / "config"
            / f"_quarto-pdf-{volume}.yml"
        )
        ordered: List[Path] = []
        if config_path.exists():
            try:
                import yaml  # type: ignore

                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                book = raw.get("book", {})
                for section in ("chapters", "appendices"):
                    for entry in book.get(section, []) or []:
                        path_str = entry.get("file", "") if isinstance(entry, dict) else entry
                        if not isinstance(path_str, str) or not path_str.endswith(".qmd"):
                            continue
                        path = (self.config_manager.book_dir / path_str).resolve()
                        if path.name in {"index.qmd", "references.qmd"}:
                            continue
                        if path.exists():
                            ordered.append(path)
            except Exception as exc:
                console.print(
                    f"[yellow]Could not parse {config_path.name}: {exc}[/yellow]"
                )

        # Include any source file under the volume that is not listed in the
        # PDF config, so audit coverage remains complete during chapter moves.
        volume_dir = self.config_manager.book_dir / "contents" / volume
        if volume_dir.exists():
            ordered.extend(sorted(volume_dir.rglob("*.qmd")))
        return self._dedupe_paths(ordered)

    @staticmethod
    def _dedupe_paths(paths: List[Path]) -> List[Path]:
        seen = set()
        out = []
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
        return out

    def _extract_pipe_tables(self, qmd_path: Path) -> List[TableAuditEntry]:
        try:
            lines = qmd_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        entries: List[TableAuditEntry] = []
        rel = self._source_rel(qmd_path)
        i = 0
        while i < len(lines) - 1:
            if not self._is_pipe_table_start(lines, i):
                i += 1
                continue

            start = i
            end = i
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1

            caption_idx = end
            while caption_idx < len(lines) and not lines[caption_idx].strip():
                caption_idx += 1
            if caption_idx >= len(lines):
                i = end
                continue

            caption_line = lines[caption_idx].strip()
            if not caption_line.startswith(":") or "#tbl-" not in caption_line:
                i = end
                continue

            table_lines = lines[start:end]
            label = self._table_label_from_caption(caption_line)
            if not label:
                i = caption_idx + 1
                continue

            columns = self._pipe_column_count(table_lines[1])
            rows = max(0, len(table_lines) - 2)
            cells = self._pipe_table_cells(table_lines)
            max_cell_chars = max((len(c) for c in cells), default=0)
            colwidths = self._table_colwidths_from_caption(caption_line)
            caption_text = self._caption_text(caption_line)

            entries.append(TableAuditEntry(
                index=0,
                label=label,
                caption=caption_text,
                source_file=rel,
                source_line=caption_idx + 1,
                qmd_path=qmd_path,
                table_markdown="\n".join(table_lines),
                caption_markdown=caption_line,
                columns=columns,
                rows=rows,
                max_cell_chars=max_cell_chars,
                has_colwidths=bool(colwidths),
                colwidths=colwidths,
                warnings=self._static_table_warnings(
                    columns,
                    rows,
                    max_cell_chars,
                    bool(colwidths),
                ),
            ))
            i = caption_idx + 1
        return entries

    def _source_rel(self, qmd_path: Path) -> str:
        try:
            return str(qmd_path.relative_to(self.config_manager.book_dir))
        except ValueError:
            try:
                return str(qmd_path.relative_to(self._repo_root()))
            except ValueError:
                return str(qmd_path)

    @classmethod
    def _is_pipe_table_start(cls, lines: List[str], idx: int) -> bool:
        return (
            idx + 1 < len(lines)
            and lines[idx].lstrip().startswith("|")
            and cls._is_pipe_separator(lines[idx + 1])
        )

    @staticmethod
    def _is_pipe_separator(line: str) -> bool:
        stripped = line.strip()
        if "|" not in stripped or "-" not in stripped:
            return False
        stripped = stripped.strip("|").strip()
        if not stripped:
            return False
        cells = [cell.strip() for cell in stripped.split("|")]
        return all(cell and set(cell) <= {"-", ":", " "} for cell in cells)

    @staticmethod
    def _pipe_column_count(separator_line: str) -> int:
        return len(separator_line.strip().strip("|").split("|"))

    @staticmethod
    def _pipe_table_cells(table_lines: List[str]) -> List[str]:
        cells: List[str] = []
        for line in table_lines:
            if LayoutCommand._is_pipe_separator(line):
                continue
            for cell in line.strip().strip("|").split("|"):
                text = re.sub(r"\s+", " ", cell).strip()
                text = re.sub(r"[*_`$\\{}[\]()<>&#]", "", text)
                if text:
                    cells.append(text)
        return cells

    @staticmethod
    def _table_label_from_caption(caption_line: str) -> str:
        match = re.search(r"\{[^}]*#(tbl-[A-Za-z0-9_-]+)[^}]*\}", caption_line)
        return match.group(1) if match else ""

    @staticmethod
    def _table_colwidths_from_caption(caption_line: str) -> str:
        match = re.search(r'tbl-colwidths=(["\'])(.*?)\1', caption_line)
        return match.group(2) if match else ""

    @staticmethod
    def _caption_text(caption_line: str) -> str:
        text = caption_line.lstrip(":").strip()
        text = re.sub(r"\s*\{[^}]*#tbl-[^}]*\}\s*$", "", text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _static_table_warnings(
        columns: int,
        rows: int,
        max_cell_chars: int,
        has_colwidths: bool,
    ) -> List[str]:
        warnings: List[str] = []
        if columns >= 4 and rows >= 4 and max_cell_chars >= 36 and not has_colwidths:
            warnings.append("wide-dense-table-without-colwidths")
        if columns <= 2 and rows <= 5 and has_colwidths:
            warnings.append("simple-table-has-colwidths-review-necessity")
        if columns >= 5 and not has_colwidths:
            warnings.append("many-columns-without-colwidths")
        return warnings

    def _write_table_audit_qmd(
        self,
        qmd_path: Path,
        entries: List[TableAuditEntry],
        volume: str,
    ) -> None:
        theme = (self.config_manager.book_dir / "tex" / f"theme-colors-{volume}.tex").resolve()
        header = (self.config_manager.book_dir / "tex" / "header-includes.tex").resolve()
        bib = (
            self.config_manager.book_dir
            / "contents"
            / volume
            / "backmatter"
            / "references.bib"
        ).resolve()

        lines = [
            "---",
            "format:",
            "  pdf:",
            "    documentclass: scrbook",
            "    classoption:",
            "      - twoside",
            "    pdf-engine: lualatex",
            "    keep-tex: true",
            "    number-sections: false",
            "    include-in-header:",
            f'      - file: "{theme}"',
            f'      - file: "{header}"',
            "execute:",
            "  enabled: false",
            "citation: true",
            "suppress-bibliography: true",
        ]
        if bib.exists():
            lines.extend(["bibliography:", f'  - "{bib}"'])
        lines.extend([
            "---",
            "",
            "\\pagestyle{empty}",
            "",
            f"`MLSysBook {volume.upper()} Table Layout Audit`",
            "",
        ])

        for entry in entries:
            if entry.index > 1:
                lines.extend(["\\clearpage", ""])
            lines.extend([
                f"`{entry.index:03d} {entry.label}`",
                "",
                f"`{entry.source_file}:{entry.source_line}`",
                "",
                entry.table_markdown,
                "",
                entry.caption_markdown,
                "",
            ])

        qmd_path.write_text("\n".join(lines), encoding="utf-8")

    def _render_table_audit_pdf(self, qmd_path: Path, render_log: Path) -> bool:
        if shutil.which("quarto") is None:
            console.print("[red]quarto not found on PATH.[/red]")
            return False
        console.print("[bold blue]Rendering table-only PDF[/bold blue] ...")
        proc = subprocess.run(
            ["quarto", "render", str(qmd_path), "--to", "pdf"],
            cwd=str(self._repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        render_log.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-20:])
            if tail:
                console.print(tail)
            return False
        return True

    @staticmethod
    def _count_overfull_hboxes(log_path: Path) -> int:
        if not log_path.exists():
            return 0
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 0
        return len(re.findall(r"Overfull \\hbox", text))

    def _measure_table_pdf(
        self,
        pdf_path: Path,
        entries: List[TableAuditEntry],
        min_width: float,
    ) -> None:
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            console.print("[yellow]pdfplumber unavailable; skipping width metrics.[/yellow]")
            return

        by_index = {entry.index: entry for entry in entries}
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                marker = TABLE_MARKER_RE.search(text[:600])
                if not marker:
                    continue
                entry = by_index.get(int(marker.group("idx")))
                if entry is None:
                    continue
                entry.warnings = [
                    warning for warning in entry.warnings
                    if warning == "simple-table-has-colwidths-review-necessity"
                ]
                width = self._page_table_rule_width(page)
                if width <= 0:
                    width = self._page_table_text_width(page)
                entry.rendered_sheet = page_idx
                entry.rendered_width_pt = width
                if width > 0:
                    entry.rendered_width_frac = width / PDF_BODY_WIDTH_PT
                    if entry.rendered_width_frac < min_width:
                        entry.warnings.append(
                            f"narrow-render:{entry.rendered_width_frac:.2f}"
                        )
                    if (
                        entry.has_colwidths
                        and (
                            (entry.columns <= 2 and entry.rows <= 5)
                            or (entry.columns <= 3 and entry.rows <= 3)
                        )
                    ):
                        entry.warnings.append("colwidths-may-be-unnecessary")

    @staticmethod
    def _page_table_rule_width(page) -> float:
        widths = []
        for line in list(getattr(page, "lines", []) or []):
            x0, x1 = line.get("x0", 0.0), line.get("x1", 0.0)
            y0, y1 = line.get("y0", 0.0), line.get("y1", 0.0)
            top = line.get("top", 0.0)
            width = x1 - x0
            if abs(y1 - y0) <= 0.5 and width >= 30 and top >= 95:
                widths.append(width)
        return max(widths, default=0.0)

    @staticmethod
    def _page_table_text_width(page) -> float:
        chars = [
            c for c in getattr(page, "chars", []) or []
            if c.get("top", 0) >= 115 and c.get("bottom", 0) <= page.height * 0.94
        ]
        if not chars:
            return 0.0
        return max(c["x1"] for c in chars) - min(c["x0"] for c in chars)

    def _make_table_contact_sheets(
        self,
        pdf_path: Path,
        out_dir: Path,
        cols: int = 3,
        rows: int = 4,
        dpi: int = 110,
    ) -> List[Path]:
        try:
            import pypdfium2 as pdfium  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError:
            console.print(
                "[yellow]pypdfium2/Pillow unavailable; skipping contact sheets.[/yellow]"
            )
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("sheet-*.png"):
            old.unlink()

        pdf = pdfium.PdfDocument(str(pdf_path))
        per_sheet = cols * rows
        cell_w, cell_h = 360, 470
        pad = 16
        label_h = 18
        sheet_w = cols * cell_w + (cols + 1) * pad
        sheet_h = rows * (cell_h + label_h) + (rows + 1) * pad
        sheets: List[Path] = []
        sheet_img = None
        draw = None

        for page_idx in range(len(pdf)):
            slot = page_idx % per_sheet
            if slot == 0:
                sheet_img = Image.new("RGB", (sheet_w, sheet_h), "white")
                draw = ImageDraw.Draw(sheet_img)

            assert sheet_img is not None and draw is not None
            page = pdf[page_idx]
            bitmap = page.render(scale=dpi / 72.0)
            thumb = bitmap.to_pil().convert("RGB")
            thumb.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

            col = slot % cols
            row = slot // cols
            x = pad + col * (cell_w + pad)
            y = pad + row * (cell_h + label_h + pad)
            draw.text((x, y), f"sheet {page_idx + 1}", fill=(80, 80, 80))
            sheet_img.paste(thumb, (x, y + label_h))
            page.close()

            if slot == per_sheet - 1 or page_idx == len(pdf) - 1:
                out = out_dir / f"sheet-{len(sheets) + 1:02d}.png"
                sheet_img.save(out)
                sheets.append(out)

        pdf.close()
        return sheets

    def _write_table_reports(
        self,
        entries: List[TableAuditEntry],
        json_path: Path,
        csv_path: Path,
    ) -> None:
        rows = [entry.to_report_dict() for entry in entries]
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "label",
                    "source_file",
                    "source_line",
                    "columns",
                    "rows",
                    "max_cell_chars",
                    "has_colwidths",
                    "colwidths",
                    "rendered_sheet",
                    "rendered_width_pt",
                    "rendered_width_frac",
                    "warnings",
                    "caption",
                ],
            )
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["warnings"] = ";".join(row["warnings"])
                writer.writerow(row)

    def _render_table_summary(
        self,
        entries: List[TableAuditEntry],
        pdf_path: Optional[Path],
        json_path: Path,
        csv_path: Path,
        min_width: float,
        overfull_count: int,
        rendered: bool,
    ) -> None:
        warned = [e for e in entries if e.warnings]
        narrow = [e for e in entries if any(w.startswith("narrow-render") for w in e.warnings)]
        explicit = [e for e in entries if e.has_colwidths]

        table = Table(show_header=True, header_style="bold")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("tables", str(len(entries)))
        table.add_row("explicit tbl-colwidths", str(len(explicit)))
        table.add_row(f"below min-width {min_width:.2f}", str(len(narrow)))
        table.add_row("tables with warnings", str(len(warned)))
        if rendered:
            table.add_row("overfull hbox warnings", str(overfull_count))
        console.print(Panel(table, title="Table Layout Audit", border_style="cyan"))
        if pdf_path is not None:
            console.print(f"[dim]PDF:[/dim] {pdf_path}")
        console.print(f"[dim]JSON:[/dim] {json_path}")
        console.print(f"[dim]CSV:[/dim] {csv_path}")

        if warned:
            console.print("[bold yellow]Top warning candidates:[/bold yellow]")
            for entry in warned[:20]:
                width = (
                    f"{entry.rendered_width_frac:.2f}"
                    if entry.rendered_width_frac
                    else "?"
                )
                console.print(
                    f"  {entry.index:03d} {entry.label} "
                    f"[dim]{entry.source_file}:{entry.source_line} "
                    f"width={width}[/dim] "
                    f"{', '.join(entry.warnings)}"
                )
            if len(warned) > 20:
                console.print(f"  [dim]...and {len(warned) - 20} more[/dim]")

    # ------------------------------------------------------------------
    # check
    # ------------------------------------------------------------------

    def _check(
        self,
        pdf_path: Path,
        threshold: float,
        limit: int,
        only: Optional[set] = None,
        skip_frontmatter: bool = False,
        csv: bool = False,
        chapter_filter: Optional[List[str]] = None,
    ) -> bool:
        if not pdf_path.exists():
            console.print(f"[red]PDF not found:[/red] {pdf_path}")
            return False

        try:
            import pdfplumber  # type: ignore
        except ImportError:
            console.print(
                "[red]pdfplumber not installed.[/red] "
                "Run: [cyan]pip install pdfplumber[/cyan]"
            )
            return False

        console.print(
            f"[bold blue]Scanning[/bold blue] {pdf_path.name} "
            f"[dim](threshold={int(threshold*100)}%, "
            f"main col <{int(MAIN_COL_RIGHT_FRAC*100)}% page width)[/dim]"
        )

        chapter_starts, labels = self._load_chapter_map(pdf_path)
        source_map = self._build_source_map(pdf_path)
        chapter_start_sheets = {start for start, _ in chapter_starts}

        flagged: List[PageReport] = []
        scanned = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            n_scan = n_pages if limit <= 0 else min(limit, n_pages)

            for i in range(n_scan):
                page = pdf.pages[i]
                next_page = pdf.pages[i + 1] if i + 1 < n_pages else None
                sheet = i + 1
                label = labels[i] if i < len(labels) else str(sheet)
                chapter = self._chapter_for(sheet, chapter_starts)
                if not self._matches_chapter(chapter, chapter_filter):
                    continue
                report = self._scan_page(page, next_page, sheet, label, chapter)
                scanned += 1
                if report and report.whitespace_frac >= threshold:
                    # Enrich with source-locator + class/action fields.
                    report.next_page_starts_chapter = (
                        (sheet + 1) in chapter_start_sheets
                    )
                    qmd_path = source_map.get(chapter)
                    if qmd_path is not None:
                        report.source_file = str(qmd_path)
                        # Resolve to absolute path for actual file reads.
                        repo_root = self._repo_root_for(pdf_path)
                        if repo_root is not None:
                            abs_path = repo_root / qmd_path
                            line_num = self._find_source_line(
                                abs_path, report.detail
                            )
                            report.source_line = line_num
                            if line_num > 0:
                                report.section = self._find_section(
                                    abs_path, line_num
                                )
                    report.klass, report.action = self._classify(report)
                    flagged.append(report)

        # Apply filters AFTER collection so the unfiltered counts are
        # available for the summary line.
        total_flagged = len(flagged)
        if skip_frontmatter:
            flagged = [r for r in flagged if not r.is_frontmatter]
        if only:
            flagged = [r for r in flagged if r.cause in only]

        if csv:
            self._render_csv(flagged)
        else:
            self._render(
                flagged,
                scanned,
                threshold,
                pdf_path,
                total_flagged=total_flagged,
            )
        return True

    # ------------------------------------------------------------------
    # collisions
    # ------------------------------------------------------------------

    def _collisions(
        self,
        pdf_path: Path,
        chapter_filter: Optional[List[str]] = None,
        csv: bool = False,
    ) -> bool:
        """Detect body content invading the header / footer band.

        The header band (top 6%) is reserved for the running header and
        page number; the footer band (bottom 6%) for the page number /
        bottom rule. Any body text whose line lands inside either band
        is a layout regression — most commonly caused by overly-tight
        tcolorbox padding or margin overrides.

        Chars sharing a y-baseline within 2pt are clustered as one
        logical line, so running-header content (page number + chapter
        title on slightly different baselines) is not double-counted.
        """
        if not pdf_path.exists():
            console.print(f"[red]PDF not found:[/red] {pdf_path}")
            return False
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            console.print(
                "[red]pdfplumber not installed.[/red] "
                "Run: [cyan]pip install pdfplumber[/cyan]"
            )
            return False

        header_collisions: List[CollisionFinding] = []
        footer_collisions: List[CollisionFinding] = []
        chapter_starts, labels = self._load_chapter_map(pdf_path)
        scanned = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                sheet = i + 1
                label = labels[i] if i < len(labels) else str(sheet)
                chapter = self._chapter_for(sheet, chapter_starts)
                if not self._matches_chapter(chapter, chapter_filter):
                    continue
                scanned += 1
                ph = page.height
                pw = page.width
                header_bottom = ph * HEADER_FRAC
                footer_top = ph * (1.0 - FOOTER_FRAC)

                # Cluster chars within 2pt of same y as one logical line.
                lines: Dict[float, list] = {}
                for c in page.chars:
                    if c["x0"] > pw * MAIN_COL_RIGHT_FRAC:
                        continue
                    placed = False
                    for ly in list(lines.keys()):
                        if abs(c["top"] - ly) < 2.0:
                            lines[ly].append(c)
                            placed = True
                            break
                    if not placed:
                        lines[c["top"]] = [c]

                ys = sorted(lines.keys())

                # Header collision: more than one logical line in header band.
                hdr = [y for y in ys if y < header_bottom + 5]
                if len(hdr) > 1:
                    snippet = self._line_text(lines[hdr[1]])
                    header_collisions.append(CollisionFinding(
                        sheet=sheet, label=label, chapter=chapter,
                        band="header", y=hdr[1], snippet=snippet,
                    ))

                # Footer collision: more than one logical line in footer band.
                ftr = [y for y in ys if y > footer_top - 5]
                if len(ftr) > 1:
                    snippet = self._line_text(lines[ftr[0]])
                    footer_collisions.append(CollisionFinding(
                        sheet=sheet, label=label, chapter=chapter,
                        band="footer", y=ftr[0], snippet=snippet,
                    ))

        if csv:
            self._render_collisions_csv(header_collisions + footer_collisions)
            return True

        if not header_collisions and not footer_collisions:
            console.print(
                Panel(
                    f"No header or footer collisions across "
                    f"{scanned} "
                    f"pages.",
                    title="✅ Collision check clean",
                    border_style="green",
                )
            )
            return True

        console.print()
        console.print(
            f"[bold yellow]⚠ Header collisions: "
            f"{len(header_collisions)}[/bold yellow]"
        )
        for item in header_collisions[:20]:
            console.print(
                f"  p.{item.label} sheet {item.sheet} {item.chapter} "
                f"y={item.y:.0f}: [dim]{item.snippet}[/dim]"
            )
        if len(header_collisions) > 20:
            console.print(
                f"  [dim]…and {len(header_collisions) - 20} more[/dim]"
            )
        console.print(
            f"[bold yellow]⚠ Footer collisions: "
            f"{len(footer_collisions)}[/bold yellow]"
        )
        for item in footer_collisions[:20]:
            console.print(
                f"  p.{item.label} sheet {item.sheet} {item.chapter} "
                f"y={item.y:.0f}: [dim]{item.snippet}[/dim]"
            )
        return True

    def _render_collisions_csv(
        self, findings: List[CollisionFinding]
    ) -> None:
        import csv as _csv
        import sys
        writer = _csv.writer(sys.stdout)
        writer.writerow(["chapter", "sheet", "label", "band", "y", "snippet"])
        for f in findings:
            writer.writerow([
                f.chapter,
                f.sheet,
                f.label,
                f.band,
                f"{f.y:.1f}",
                f.snippet,
            ])

    @staticmethod
    def _line_text(line_chars: list) -> str:
        line_chars = sorted(line_chars, key=lambda c: c["x0"])
        return "".join(c.get("text", "") for c in line_chars).strip()[:60]

    # ------------------------------------------------------------------
    # margins — margin figure / note overflow off the page bottom
    # ------------------------------------------------------------------

    def _margins(
        self,
        pdf_path: Path,
        tol: float = 2.0,
        limit: int = 0,
        csv: bool = False,
        chapter_filter: Optional[List[str]] = None,
        include_overlaps: bool = False,
    ) -> bool:
        """Flag margin-column content (figures, captions, notes) that runs
        below the page's usable bottom — the documented ``figure-margin.md``
        §7.2 failure mode where a margin figure anchored near a page break has
        nowhere to float and clips off the page.

        Returns True when clean, False when any overflow is found, so
        ``binder layout margins`` exits non-zero as an explicit gate.
        """
        if not pdf_path.exists():
            console.print(f"[red]PDF not found:[/red] {pdf_path}")
            return False
        try:
            import pdfplumber  # type: ignore  # noqa: F401
        except ImportError:
            console.print(
                "[red]pdfplumber not installed.[/red] "
                "Run: [cyan]pip install pdfplumber[/cyan]"
            )
            return False

        console.print(
            f"[bold blue]Scanning margins[/bold blue] {pdf_path.name} "
            f"[dim](margin col <{int(LEFT_MARGIN_RIGHT_FRAC*100)}% "
            f"or >{int(RIGHT_MARGIN_LEFT_FRAC*100)}% width, "
            f"overflow = below {int((1-FOOTER_FRAC)*100)}% page height "
            f"+ {tol:.0f}pt)[/dim]"
        )
        findings, scanned = self._collect_margin_findings(
            pdf_path,
            tol,
            limit,
            chapter_filter=chapter_filter,
            include_overlaps=include_overlaps,
        )
        if csv:
            self._render_margins_csv(findings)
        else:
            self._render_margins(findings, scanned, pdf_path)
        return not any(f.severity == "error" for f in findings)

    @staticmethod
    def _margin_side(pw, x0, x1) -> str:
        """Return the physical margin side containing an object, if any."""
        if x1 <= pw * LEFT_MARGIN_RIGHT_FRAC:
            return "left"
        if x0 >= pw * RIGHT_MARGIN_LEFT_FRAC:
            return "right"
        return ""

    @classmethod
    def _margin_graphic_objects(cls, pw, images, rects=None, curves=None):
        """Return substantial image/vector boxes that sit in a margin band."""
        objects = list(images or []) + list(rects or []) + list(curves or [])
        margin_max_w = pw * 0.25
        graphics = []
        for obj in objects:
            x0 = obj.get("x0", 0)
            x1 = obj.get("x1", 0)
            top = obj.get("top", 0)
            bottom = obj.get("bottom", 0)
            width = x1 - x0
            height = bottom - top
            if not cls._margin_side(pw, x0, x1):
                continue
            if width <= 0 or height <= 0 or width > margin_max_w:
                continue
            graphics.append(obj)
        return graphics

    @classmethod
    def _page_overflow(cls, pw, ph, chars, images, tol):
        """Pure geometry: true margin-column chars/images whose bottom runs
        below the usable bottom (footer band top + tol). Returns (over_chars,
        over_imgs). Coordinates follow pdfplumber: x0 from left, bottom from
        page top (increasing downward). Unit-tested with dict fixtures.

        A char counts as margin content when it sits in either physical margin
        band. This handles bound PDFs where margin notes alternate left/right.
        Likewise an image/vector box must be narrow and fully in a margin band,
        not a full-text-block figure straddling the body column.
        """
        usable_bottom = ph * (1.0 - FOOTER_FRAC) + tol

        # Text overflow is bounded to the page: a real margin caption clips at
        # the page edge (dips into the footer band, still on-page). Text flung
        # far BELOW the page edge (bottom > ph) is figure-internal label text
        # mispositioned off-canvas, not a margin caption — exclude it. Images
        # (below) stay uncapped: a tall margin figure legitimately runs off.
        over_chars = [
            line for line in cls._margin_text_lines(pw, chars)
            if usable_bottom < line["bottom"] <= ph
        ]
        margin_max_w = pw * 0.25
        over_imgs = [
            im for im in images
            if cls._margin_side(pw, im.get("x0", 0), im.get("x1", 0))
            and (im.get("x1", 0) - im.get("x0", 0)) <= margin_max_w
            and im["bottom"] > usable_bottom
        ]
        return over_chars, over_imgs

    @classmethod
    def _margin_text_lines(cls, pw, chars) -> List[dict]:
        """Cluster left/right margin chars into logical text-line boxes."""
        raw_lines: Dict[Tuple[str, float], list] = {}
        for c in chars:
            side = cls._margin_side(pw, c["x0"], c["x1"])
            if not side:
                continue
            for key in raw_lines:
                key_side, ly = key
                if key_side == side and abs(c["top"] - ly) < 2.0:
                    raw_lines[key].append(c)
                    break
            else:
                raw_lines[(side, c["top"])] = [c]

        lines = []
        for key in sorted(raw_lines, key=lambda item: (item[0], item[1])):
            line_chars = sorted(raw_lines[key], key=lambda c: c["x0"])
            text = "".join(ch.get("text", "") for ch in line_chars).strip()
            if not text:
                continue
            lines.append({
                "side": key[0],
                "x0": min(ch["x0"] for ch in line_chars),
                "x1": max(ch["x1"] for ch in line_chars),
                "top": min(ch["top"] for ch in line_chars),
                "bottom": max(ch["bottom"] for ch in line_chars),
                "text": text[:90],
            })
        return lines

    @staticmethod
    def _substantial_margin_text(text: str) -> bool:
        return len(re.findall(r"[A-Za-z0-9]", text or "")) >= 6

    @classmethod
    def _margin_baseline_crowding(
        cls,
        pw,
        chars,
        min_gap: float = 3.0,
        min_deficit: float = 1.0,
    ) -> List[Tuple[dict, dict, float]]:
        """Return adjacent margin text lines whose baselines are implausibly close.

        This intentionally uses baseline/top distance instead of bbox
        intersection. Normal sidenote leading can make adjacent line boxes
        geometrically overlap; truly bad collisions have baselines only a few
        points apart.
        """
        lines = cls._margin_text_lines(pw, chars)
        findings: List[Tuple[dict, dict, float]] = []
        for prev, cur in zip(lines, lines[1:]):
            if prev.get("side") != cur.get("side"):
                continue
            if not (
                cls._substantial_margin_text(prev["text"])
                and cls._substantial_margin_text(cur["text"])
            ):
                continue
            baseline_gap = cur["top"] - prev["top"]
            deficit = min_gap - baseline_gap
            if 0 < baseline_gap < min_gap and deficit >= min_deficit:
                findings.append((prev, cur, deficit))
        return findings

    @classmethod
    def _margin_image_text_overlaps(
        cls,
        pw,
        chars,
        images,
        min_image_edge: float = 20.0,
    ) -> List[Tuple[dict, dict, float]]:
        """Return substantial margin text lines intersecting substantial images.

        Tiny icons are ignored; those often sit inside callout artwork and are
        expected to share a band with labels.
        """
        lines = cls._margin_text_lines(pw, chars)
        overlaps: List[Tuple[dict, dict, float]] = []
        for im in images or []:
            im_side = cls._margin_side(pw, im.get("x0", 0), im.get("x1", 0))
            if not im_side:
                continue
            width = im.get("x1", 0) - im.get("x0", 0)
            height = im.get("bottom", 0) - im.get("top", 0)
            if width < min_image_edge or height < min_image_edge:
                continue
            for line in lines:
                if line.get("side") != im_side:
                    continue
                if not cls._substantial_margin_text(line["text"]):
                    continue
                x_overlap = min(line["x1"], im["x1"]) - max(line["x0"], im["x0"])
                y_overlap = min(line["bottom"], im["bottom"]) - max(line["top"], im["top"])
                if x_overlap > 0 and y_overlap > 1.0:
                    overlaps.append((line, im, y_overlap))
        return overlaps

    def _collect_margin_findings(
        self,
        pdf_path: Path,
        tol: float = 2.0,
        limit: int = 0,
        chapter_filter: Optional[List[str]] = None,
        include_overlaps: bool = False,
    ) -> Tuple[List["MarginFinding"], int]:
        """Detection core (no console output). Returns (findings, pages_scanned).

        Shared by the ``binder layout margins`` gate and the warn-only build
        postflight (via the module-level ``scan_margin_overflow`` helper).
        """
        import pdfplumber  # type: ignore

        chapter_starts, labels = self._load_chapter_map(pdf_path)
        source_map = self._build_source_map(pdf_path)

        # Resolve repo root once for source-line lookups.
        repo_root = self._repo_root_for(pdf_path)

        findings: List[MarginFinding] = []
        scanned = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            n_scan = n_pages if limit <= 0 else min(limit, n_pages)
            for i in range(n_scan):
                page = pdf.pages[i]
                sheet = i + 1
                pw, ph = page.width, page.height
                footer_top = ph * (1.0 - FOOTER_FRAC)
                chapter = self._chapter_for(sheet, chapter_starts)
                if not self._matches_chapter(chapter, chapter_filter):
                    continue
                scanned += 1
                label = labels[i] if i < len(labels) else str(sheet)

                margin_text_lines = self._margin_text_lines(pw, page.chars)
                line_texts = [line["text"] for line in margin_text_lines]
                margin_graphics = self._margin_graphic_objects(
                    pw, page.images, page.rects, page.curves
                )

                over_chars, over_imgs = self._page_overflow(
                    pw, ph, page.chars, margin_graphics, tol
                )

                if over_chars or over_imgs:
                    bottoms = (
                        [c["bottom"] for c in over_chars]
                        + [im["bottom"] for im in over_imgs]
                    )
                    max_bottom = max(bottoms)
                    over_pts = max_bottom - footer_top
                    off_page = max_bottom > ph
                    sig = []
                    if over_imgs:
                        sig.append("image")
                    if over_chars:
                        sig.append("caption/text")
                    signal = "+".join(sig)
                    snippet = line_texts[-1] if line_texts else ""
                    src_file, src_line, section = self._locate_margin_source(
                        repo_root, source_map, chapter, line_texts
                    )
                    findings.append(MarginFinding(
                        sheet=sheet, label=label, chapter=chapter,
                        signal=signal, over_pts=over_pts, off_page=off_page,
                        snippet=snippet, source_file=src_file,
                        source_line=src_line, section=section,
                    ))

                if include_overlaps:
                    crowding = self._margin_baseline_crowding(pw, page.chars)
                    if crowding:
                        prev, cur, deficit = max(
                            crowding, key=lambda item: item[2]
                        )
                        overlap_lines = [prev["text"], cur["text"]]
                        src_file, src_line, section = self._locate_margin_source(
                            repo_root, source_map, chapter, overlap_lines
                        )
                        findings.append(MarginFinding(
                            sheet=sheet, label=label, chapter=chapter,
                            signal="caption/text", over_pts=deficit,
                            off_page=False, snippet=cur["text"],
                            source_file=src_file, source_line=src_line,
                            section=section, issue="baseline-crowding",
                            severity="warning", related=prev["text"],
                        ))

                    image_overlaps = self._margin_image_text_overlaps(
                        pw, page.chars, margin_graphics
                    )
                    if image_overlaps:
                        line, im, y_overlap = max(
                            image_overlaps, key=lambda item: item[2]
                        )
                        src_file, src_line, section = self._locate_margin_source(
                            repo_root, source_map, chapter, [line["text"]]
                        )
                        if src_line <= 0:
                            continue
                        related = (
                            f"image {im.get('x0', 0):.0f},"
                            f"{im.get('top', 0):.0f}-"
                            f"{im.get('bottom', 0):.0f}"
                        )
                        findings.append(MarginFinding(
                            sheet=sheet, label=label, chapter=chapter,
                            signal="image+caption/text", over_pts=y_overlap,
                            off_page=False, snippet=line["text"],
                            source_file=src_file, source_line=src_line,
                            section=section, issue="image-text-overlap",
                            severity="warning", related=related,
                        ))

        return findings, scanned

    def _locate_margin_source(
        self,
        repo_root: Optional[Path],
        source_map: Dict[str, Path],
        chapter: str,
        line_texts: List[str],
    ) -> Tuple[str, int, str]:
        qmd = source_map.get(chapter)
        if qmd is None:
            return "", 0, ""
        src_file, src_line, section = str(qmd), 0, ""
        if repo_root is not None and line_texts:
            abs_path = repo_root / qmd
            # Try the longest margin lines first — caption text matches the
            # source line more reliably than a short wrapped fragment.
            for cand in sorted(line_texts, key=len, reverse=True):
                src_line = self._find_source_line(abs_path, cand)
                if src_line > 0:
                    break
            if src_line > 0:
                section = self._find_section(abs_path, src_line)
        return src_file, src_line, section

    def _render_margins(
        self, findings: List[MarginFinding], scanned: int, pdf_path: Path
    ) -> None:
        if not findings:
            console.print(Panel(
                f"No margin overflow across {scanned} pages — every margin "
                f"figure/note sits on-page.",
                title="✅ Margin check clean",
                border_style="green",
            ))
            return
        errors = [f for f in findings if f.severity == "error"]
        warnings = [f for f in findings if f.severity != "error"]
        console.print()
        if errors:
            console.print(
                f"[bold red]✗ {len(errors)} blocking margin overflow"
                f"{'s' if len(errors) != 1 else ''}[/bold red] "
                f"[dim]+ {len(warnings)} warning candidate"
                f"{'s' if len(warnings) != 1 else ''}[/dim]"
            )
        else:
            console.print(
                f"[bold yellow]⚠ {len(warnings)} margin warning candidate"
                f"{'s' if len(warnings) != 1 else ''}[/bold yellow] "
                f"[dim](no blocking footer/off-page overflow)[/dim]"
            )
        table = Table(show_header=True, header_style="bold")
        table.add_column("page", justify="right")
        table.add_column("chapter")
        table.add_column("issue")
        table.add_column("sev")
        table.add_column("signal")
        table.add_column("pts", justify="right")
        table.add_column("source", overflow="fold")
        for f in sorted(findings, key=lambda r: (r.severity != "error", -r.over_pts)):
            over = f"{f.over_pts:.0f}pt"
            if f.off_page:
                over += " ⎋"  # off the physical page edge
            src = f.source_file
            if f.source_line:
                src += f":{f.source_line}"
            if f.section:
                src += f"  [{f.section}]"
            table.add_row(
                f"{f.label} (sheet {f.sheet})",
                f.chapter,
                f.issue,
                f.severity,
                f.signal,
                over,
                src or "[dim]?[/dim]",
            )
        console.print(table)
        console.print(
            "[dim]⎋ = past the physical page edge. Fix per figure-margin.md "
            "§7: reposition the .column-margin block to a footnote-sparse, "
            "mid-page anchor, shorten the caption, or cut it (§0 gate). "
            "Overlap/crowding rows are warning-level candidates and require "
            "visual confirmation.[/dim]"
        )

    def _render_margins_csv(self, findings: List[MarginFinding]) -> None:
        import csv as _csv
        import sys
        writer = _csv.writer(sys.stdout)
        writer.writerow([
            "chapter", "sheet", "label", "issue", "severity", "signal",
            "over_pts", "off_page", "source_file", "source_line", "section",
            "snippet", "related",
        ])
        for f in findings:
            writer.writerow([
                f.chapter, f.sheet, f.label, f.issue, f.severity, f.signal,
                f"{f.over_pts:.1f}", int(f.off_page), f.source_file,
                f.source_line, f.section, f.snippet, f.related,
            ])

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _render_csv(self, flagged: List[PageReport]) -> None:
        import csv as _csv
        import sys
        writer = _csv.writer(sys.stdout)
        writer.writerow([
            "chapter", "sheet", "label", "gap_pct", "pts",
            "culprit", "detail", "size_hint", "fix_hint", "is_frontmatter",
            "source_file", "source_line", "section",
            "next_page_starts_chapter", "class", "action",
        ])
        for r in flagged:
            writer.writerow([
                r.chapter,
                r.sheet,
                r.label,
                int(r.whitespace_frac * 100),
                int(r.whitespace_pts),
                r.cause,
                r.detail,
                r.size_hint,
                r.fix_hint,
                int(r.is_frontmatter),
                r.source_file,
                r.source_line,
                r.section,
                int(r.next_page_starts_chapter),
                r.klass,
                r.action,
            ])

    # ------------------------------------------------------------------
    # outline + page labels
    # ------------------------------------------------------------------

    def _load_chapter_map(
        self, pdf_path: Path
    ) -> Tuple[List[Tuple[int, str]], List[str]]:
        """Return (chapter_starts, labels).

        chapter_starts: [(start_sheet_1indexed, title)] for top-level outline
            entries only. Sorted ascending by start sheet.
        labels: list of printed page labels indexed by 0-based page index,
            same length as pdf.pages. Falls back to the 1-based sheet number
            when PageLabels are missing.
        """
        try:
            import pypdf  # type: ignore
        except ImportError:
            return [], []

        try:
            r = pypdf.PdfReader(str(pdf_path))
        except Exception:
            return [], []

        n = len(r.pages)

        # Printed page labels (PDF /PageLabels). Pad to len(pages).
        try:
            labels = list(r.page_labels or [])
        except Exception:
            labels = []
        if len(labels) < n:
            labels.extend(str(i + 1) for i in range(len(labels), n))

        # Top-level outline entries only — these are chapters in this book.
        chapter_starts: List[Tuple[int, str]] = []
        try:
            outline = r.outline or []
        except Exception:
            outline = []
        for item in outline:
            if isinstance(item, list):
                continue  # skip nested sub-entries
            try:
                page0 = r.get_destination_page_number(item)
                title = getattr(item, "title", None) or "(untitled)"
                chapter_starts.append((page0 + 1, title))
            except Exception:
                continue
        chapter_starts.sort(key=lambda x: x[0])
        return chapter_starts, labels

    def _chapter_for(
        self, sheet_1indexed: int, chapter_starts: List[Tuple[int, str]]
    ) -> str:
        """Return the title of the chapter whose range contains this sheet."""
        if not chapter_starts:
            return "(no outline)"
        current = "Frontmatter"
        for start, title in chapter_starts:
            if start <= sheet_1indexed:
                current = title
            else:
                break
        return current

    # ------------------------------------------------------------------
    # source map: chapter title → QMD file
    # ------------------------------------------------------------------

    def _build_source_map(self, pdf_path: Path) -> Dict[str, Path]:
        """Return {chapter_title → QMD path}, scoped to the PDF volume when known.

        Vol I and Vol II intentionally share chapter titles such as
        "Introduction". Source localization must therefore scan only the
        matching ``contents/volN`` tree when the PDF path/name identifies a
        volume.
        """
        # Find repo root by walking up from pdf_path until we hit a dir
        # containing book/quarto/contents.
        repo_root = self._repo_root_for(pdf_path)
        if repo_root is None:
            return {}

        contents = repo_root / "book" / "quarto" / "contents"
        volume = self._volume_from_pdf_path(pdf_path)
        if volume and (contents / volume).is_dir():
            vol_dirs = [contents / volume]
        else:
            vol_dirs = sorted(contents.glob("vol*"))
        out: Dict[str, Path] = {}
        for vol_dir in vol_dirs:
            for chapter_dir in sorted(vol_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                qmd = chapter_dir / f"{chapter_dir.name}.qmd"
                if not qmd.exists():
                    continue
                # Read first H1 from the file.
                try:
                    with qmd.open() as f:
                        for line in f:
                            line = line.rstrip("\n")
                            if line.startswith("# ") and not line.startswith("##"):
                                # Strip "{#sec-…}" attribute suffix.
                                title = line[2:].strip()
                                hash_idx = title.find("{#")
                                if hash_idx != -1:
                                    title = title[:hash_idx].rstrip()
                                out[title] = qmd.relative_to(repo_root)
                                break
                except OSError:
                    continue
        return out

    @staticmethod
    def _repo_root_for(pdf_path: Path) -> Optional[Path]:
        cur = Path(pdf_path).resolve().parent
        for _ in range(8):
            if (cur / "book" / "quarto" / "contents").is_dir():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return None

    @staticmethod
    def _volume_from_pdf_path(pdf_path: Path) -> Optional[str]:
        haystack = " ".join([Path(pdf_path).name, *Path(pdf_path).parts]).lower()
        if "vol1" in haystack:
            return "vol1"
        if "vol2" in haystack:
            return "vol2"
        return None

    @staticmethod
    def _parse_chapter_filter(raw: str) -> List[str]:
        return [part.strip() for part in (raw or "").split(",") if part.strip()]

    @staticmethod
    def _chapter_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @classmethod
    def _matches_chapter(
        cls,
        chapter: str,
        chapter_filter: Optional[List[str]],
    ) -> bool:
        if not chapter_filter:
            return True
        chapter_key = cls._chapter_key(chapter)
        chapter_slug = chapter_key.replace(" ", "-")
        for raw in chapter_filter:
            wanted_key = cls._chapter_key(raw)
            wanted_slug = wanted_key.replace(" ", "-")
            if not wanted_key:
                continue
            if wanted_key == chapter_key or wanted_slug == chapter_slug:
                return True
            if wanted_key in chapter_key:
                return True
        return False

    @staticmethod
    def _find_source_line(qmd_path: Path, detail: str) -> int:
        """Find a line in the QMD file matching the detail text.

        pdfplumber sometimes strips spaces in extracted text, so we
        normalize both source and detail by removing all whitespace,
        then look for the detail as a substring of any source line's
        normalized form. Returns 1-indexed line number or 0.
        """
        if not detail or not qmd_path.exists():
            return 0
        needle = "".join(detail.split()).lower()
        if len(needle) < 12:
            return 0  # too short to match reliably
        # Compare leading 40 chars to avoid spurious matches.
        needle = needle[:40]
        try:
            with qmd_path.open() as f:
                for i, raw in enumerate(f, start=1):
                    haystack = "".join(raw.split()).lower()
                    if needle and needle in haystack:
                        return i
        except OSError:
            pass
        return 0

    @staticmethod
    def _find_section(qmd_path: Path, line_num: int) -> str:
        """Walk backwards from line_num to find the nearest H2/H3.

        Returns the header text without the markdown ## / ### prefix,
        with any trailing `{#sec-…}` attribute stripped.
        """
        if not qmd_path.exists() or line_num <= 0:
            return ""
        try:
            with qmd_path.open() as f:
                lines = f.readlines()
        except OSError:
            return ""
        start = min(line_num, len(lines)) - 1
        for i in range(start, -1, -1):
            line = lines[i].rstrip("\n")
            m = None
            if line.startswith("### "):
                m = line[4:]
            elif line.startswith("## "):
                m = line[3:]
            if m is not None:
                hash_idx = m.find("{#")
                if hash_idx != -1:
                    m = m[:hash_idx]
                return m.strip()
        return ""

    @staticmethod
    def _classify(report: "PageReport") -> Tuple[str, str]:
        """Return (class, action) for a flagged page.

        Class definitions:
          A — Likely movable (callout/table/figure overshoot)
          B — Accept or pattern-α split (very long block)
          C — Chapter-end natural whitespace (next page starts chapter)
          D — Heading orphan-prevention (structural)
        """
        if report.next_page_starts_chapter:
            return ("C", "accept-chapter-end")
        if report.cause == "heading":
            return ("D", "accept-orphan")
        if report.cause == "end-of-document":
            return ("C", "filter-end-of-doc")
        if report.cause in ("table", "figure", "callout/box"):
            # Big gaps are likely too large to be movable.
            if report.whitespace_frac >= 0.55:
                return ("B", "accept-or-split")
            return ("A", "try-move-up")
        if report.cause == "paragraph":
            return ("A", "try-move-up")
        return ("?", "manual-review")

    # ------------------------------------------------------------------
    # per-page analysis
    # ------------------------------------------------------------------

    def _scan_page(
        self, page, next_page, sheet: int, label: str, chapter: str
    ) -> Optional[PageReport]:
        chars = page.chars
        if not chars:
            return None

        page_h = page.height
        page_w = page.width

        header_y = page_h * HEADER_FRAC          # top band
        footer_y = page_h * (1.0 - FOOTER_FRAC)  # bottom band
        main_col_max_x = page_w * MAIN_COL_RIGHT_FRAC

        # Main-column chars only: exclude margin notes (right side) and
        # header/footer bands.
        body_chars = [
            c for c in chars
            if c["x0"] < main_col_max_x
            and c["top"] > header_y
            and c["bottom"] < footer_y
        ]
        if not body_chars:
            return None

        body_bottom = max(c["bottom"] for c in body_chars)
        body_top = min(c["top"] for c in body_chars)

        usable_height = footer_y - header_y
        whitespace_pts = footer_y - body_bottom
        # Normalize against usable column height, not full page.
        whitespace_frac = whitespace_pts / usable_height if usable_height > 0 else 0.0

        if next_page is None:
            cause, detail, size_hint = "end-of-document", "", ""
        else:
            cause, detail, size_hint = self._guess_cause(next_page)

        return PageReport(
            sheet=sheet,
            label=label,
            chapter=chapter,
            whitespace_pts=whitespace_pts,
            whitespace_frac=whitespace_frac,
            body_bottom_y=body_bottom,
            usable_bottom_y=footer_y,
            cause=cause,
            detail=detail,
            size_hint=size_hint,
            fix_hint=self._fix_hint(cause, size_hint),
            is_frontmatter=self._is_frontmatter_label(label),
        )

    @staticmethod
    def _is_frontmatter_label(label: str) -> bool:
        """Detect roman-numeral printed page labels (frontmatter)."""
        if not label:
            return False
        s = label.strip().lower()
        return bool(s) and all(c in "ivxlcdm" for c in s)

    @staticmethod
    def _fix_hint(cause: str, size_hint: str) -> str:
        """One-word suggested fix per culprit class."""
        if cause == "table":
            # Extract row count from size_hint if present.
            rows = 0
            if "rows" in size_hint:
                try:
                    rows = int(size_hint.split()[0])
                except (ValueError, IndexError):
                    rows = 0
            if rows >= 6:
                return "longtable?"
            if rows > 0:
                return "shrink/keep"
            return "longtable?"
        if cause == "figure":
            return "fig-pos?"
        if cause == "heading":
            return "relax-penalty?"
        if cause == "paragraph":
            return "rewrite"
        if cause == "callout/box":
            return "audit"
        if cause == "end-of-document":
            return "ignore"
        return "manual"

    # ------------------------------------------------------------------
    # cause guess: look at top of next page
    # ------------------------------------------------------------------

    def _guess_cause(self, page) -> Tuple[str, str, str]:
        """Best-effort guess at the block that probably forced the break.

        Inspects the top region of the next page (after the header band)
        and returns (cause, detail, size_hint):
            cause      — one of: table, figure, callout/box, heading,
                         paragraph, unknown
            detail     — caption / heading / title text (truncated)
            size_hint  — "10 rows", "380pt tall", etc., where extractable
        """
        if page is None:
            return ("end-of-document", "", "")

        page_h = page.height
        page_w = page.width
        header_y = page_h * HEADER_FRAC
        # "Top of next page" = first ~40% of usable height after header.
        zone_bottom = header_y + (page_h - header_y) * 0.40

        chars = [c for c in page.chars if header_y < c["top"] < zone_bottom]
        if not chars:
            return ("unknown", "", "")

        # 1. Figure? — any image object whose top is in the zone.
        images = getattr(page, "images", []) or []
        for im in images:
            if header_y < im.get("top", 0) < zone_bottom:
                im_h = im.get("height", 0) or (
                    im.get("bottom", 0) - im.get("top", 0)
                )
                size_hint = f"{int(im_h)}pt tall" if im_h else ""
                detail = self._extract_caption(page, prefix_pattern="Figure")
                return ("figure", detail, size_hint)

        # 2. Table? — many horizontal lines (rules) clustered in the zone.
        #    Tables in this book are typed with booktabs-style rules.
        lines = getattr(page, "lines", []) or []
        rects = getattr(page, "rects", []) or []
        horiz_lines_full = [
            ln for ln in lines
            if header_y < ln.get("top", 0) < zone_bottom
            and abs(ln.get("height", 0)) < 1.0
        ]
        # booktabs tables typically have >= 3 horizontal rules.
        if len(horiz_lines_full) >= 3:
            # Row estimate: count distinct text-line y-positions BETWEEN the
            # topmost and bottommost rule of the table. Booktabs uses only
            # 3 rules (top / mid / bottom) regardless of row count, so we
            # have to count baselines, not rules.
            rule_ys = sorted(ln.get("top", 0) for ln in horiz_lines_full)
            top_rule_y = rule_ys[0]
            bot_rule_y = rule_ys[-1]
            # Allow the table to extend somewhat below the zone (tables
            # often run further down the page); scan all chars within the
            # rule band.
            table_chars = [
                c for c in page.chars
                if top_rule_y - 2 <= c["top"] <= bot_rule_y + 2
            ]
            # Distinct baselines (y rounded to nearest pt).
            ys = {round(c["top"], 0) for c in table_chars}
            rows_est = max(0, len(ys) - 1)  # -1 for header row
            size_hint = f"{rows_est} rows" if rows_est else ""
            detail = self._extract_caption(page, prefix_pattern="Table")
            return ("table", detail, size_hint)

        # 3. Callout / box? — a large filled rectangle in the zone.
        #    Heuristic: a rect taller than ~30pt and at least 60% of main
        #    column width counts as a box.
        for r in rects:
            top = r.get("top", 0)
            if not (header_y < top < zone_bottom):
                continue
            height = r.get("height", 0) or (r.get("bottom", 0) - top)
            width = r.get("width", 0) or (r.get("x1", 0) - r.get("x0", 0))
            if height >= 30 and width >= page_w * 0.45:
                # First line of text inside the box = callout title.
                title = self._first_line_text(chars, top_y_min=top)
                return ("callout/box", title, "")

        # 4. Heading? — first line's font size is notably above median.
        sizes = [round(c.get("size", 0), 1) for c in chars if c.get("size")]
        if sizes:
            sorted_sizes = sorted(sizes)
            median = sorted_sizes[len(sorted_sizes) // 2]
            # First line: chars sharing the topmost band (within 2pt).
            top_y = min(c["top"] for c in chars)
            first_line_sizes = [
                round(c.get("size", 0), 1)
                for c in chars
                if abs(c["top"] - top_y) < 2.0 and c.get("size")
            ]
            if first_line_sizes:
                first_max = max(first_line_sizes)
                if first_max >= median + 1.5:
                    heading = self._first_line_text(chars)
                    return ("heading", heading, "")

        # 5. Paragraph fallback — return the first ~60 chars so it's
        # easy to grep back to the source.
        first_text = self._first_line_text(chars)
        return ("paragraph", first_text, "")

    # ------------------------------------------------------------------
    # text extraction helpers
    # ------------------------------------------------------------------

    def _first_line_text(self, chars, top_y_min: float = 0.0) -> str:
        """Reconstruct the topmost text line from a list of page chars."""
        if not chars:
            return ""
        candidates = [c for c in chars if c["top"] >= top_y_min]
        if not candidates:
            return ""
        top_y = min(c["top"] for c in candidates)
        line_chars = [c for c in candidates if abs(c["top"] - top_y) < 2.0]
        line_chars.sort(key=lambda c: c["x0"])
        text = "".join(c.get("text", "") for c in line_chars).strip()
        return self._truncate(text, 70)

    def _extract_caption(self, page, prefix_pattern: str) -> str:
        """Find a 'Table X.Y:' or 'Figure X.Y:' caption on a page.

        Captions in this book are rendered by Quarto as:
            Table N.M: Some caption text
        We extract the line beginning with the given prefix and return
        the caption text (truncated). Returns "" if not found.
        """
        try:
            text = page.extract_text() or ""
        except Exception:
            return ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line.startswith(prefix_pattern):
                continue
            # Strip "Table N.M:" / "Figure N.M:" prefix. Allow up to 30
            # chars because pdfplumber sometimes drops spaces, producing
            # e.g. "Table5.3summarizes:" instead of "Table 5.3 summarizes:".
            colon = line.find(":")
            if colon != -1 and colon < 30:
                line = line[colon + 1:].strip()
            return self._truncate(line, 70)
        return ""

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = s.replace("­", "").strip()  # strip soft hyphens
        if len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        flagged: List[PageReport],
        scanned: int,
        threshold: float,
        pdf_path: Path,
        total_flagged: int = 0,
    ) -> None:
        if not flagged:
            extra = ""
            if total_flagged and total_flagged != len(flagged):
                extra = (
                    f" (filtered from {total_flagged} unfiltered flags)"
                )
            console.print(
                Panel(
                    f"No pages exceeded {int(threshold*100)}% bottom whitespace "
                    f"across {scanned} pages{extra}.",
                    title="✅ Layout check clean",
                    border_style="green",
                )
            )
            return

        # Group by chapter; sort each group DESCENDING by sheet so the
        # natural read order matches the fix order (back-to-front within
        # a chapter, so earlier fixes don't shift later diagnoses).
        by_chapter: Dict[str, List[PageReport]] = defaultdict(list)
        for r in flagged:
            by_chapter[r.chapter].append(r)
        # Chapter order: by first-flagged sheet ascending, so the report
        # reads forward through the book even though each chapter is
        # internally descending.
        chapter_order = sorted(
            by_chapter.keys(),
            key=lambda c: min(r.sheet for r in by_chapter[c]),
        )

        cause_counts = Counter(r.cause for r in flagged)
        summary = ", ".join(f"{n}× {c}" for c, n in cause_counts.most_common())

        console.print()
        filter_note = ""
        if total_flagged and total_flagged != len(flagged):
            filter_note = (
                f" [dim](filtered from {total_flagged} total)[/dim]"
            )
        console.print(
            f"[bold yellow]⚠ {len(flagged)}/{scanned} pages flagged[/bold yellow]"
            f"{filter_note} "
            f"[dim](≥{int(threshold*100)}% bottom whitespace · "
            f"fix order: back-to-front within chapter)[/dim]"
        )
        console.print(f"[dim]by culprit: {summary}[/dim]")
        fix_counts = Counter(r.fix_hint for r in flagged)
        fix_summary = ", ".join(
            f"{n}× {h}" for h, n in fix_counts.most_common()
        )
        console.print(f"[dim]by fix hint: {fix_summary}[/dim]")

        for chapter in chapter_order:
            rows = sorted(by_chapter[chapter], key=lambda x: -x.sheet)
            console.print()
            console.print(
                Rule(
                    f"[bold magenta]{chapter}[/bold magenta] "
                    f"[dim]({len(rows)} page{'s' if len(rows) != 1 else ''})[/dim]",
                    style="magenta",
                    align="left",
                )
            )

            table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            table.add_column("p.", justify="right", style="cyan", width=5)
            table.add_column("sheet", justify="right", style="dim", width=6)
            table.add_column("gap%", justify="right", style="yellow", width=6)
            table.add_column("cls", justify="center", style="magenta", width=3)
            table.add_column("culprit", style="white", width=11)
            table.add_column("detail / source / section", style="white")
            table.add_column("action", style="green", width=20)

            for r in rows:
                label = r.label
                if r.is_frontmatter:
                    label = f"{label}*"
                # Build the multi-line detail/source/section block.
                lines = []
                if r.detail:
                    lines.append(r.detail)
                if r.source_file:
                    src = r.source_file
                    if r.source_line:
                        src = f"{src}:{r.source_line}"
                    lines.append(f"[dim]→ {src}[/dim]")
                if r.section:
                    lines.append(f"[dim]   in: {r.section}[/dim]")
                if not lines:
                    lines.append(r.size_hint or "")
                detail_block = "\n".join(lines)
                table.add_row(
                    label,
                    str(r.sheet),
                    f"{int(r.whitespace_frac * 100)}%",
                    r.klass or "?",
                    r.cause,
                    detail_block,
                    r.action or r.fix_hint,
                )
            console.print(table)

        console.print()
        console.print(
            f"[dim]PDF:[/dim] {pdf_path}\n"
            f"[dim]Within each chapter, fix the LAST flagged page first — "
            f"front-to-back fixes shift downstream pages and invalidate "
            f"later diagnoses.[/dim]\n"
            f"[dim]Asterisk (*) after p. = roman-numeral label "
            f"(frontmatter/backmatter — usually intentional). "
            f"Use --skip-frontmatter to hide.[/dim]"
        )


def scan_margin_overflow(pdf_path: Path, tol: float = 2.0, limit: int = 0):
    """Module-level entry point for warn-only margin-overflow detection.

    Returns the list of MarginFinding objects (empty when clean), or None when
    detection cannot run (missing PDF or pdfplumber). Used by the PDF build
    postflight to surface margin overflow as a non-blocking warning; the
    blocking gate is ``binder layout margins`` (LayoutCommand._margins).
    """
    if not Path(pdf_path).exists():
        return None
    try:
        import pdfplumber  # type: ignore  # noqa: F401
    except ImportError:
        return None
    cmd = LayoutCommand(None, None)
    findings, _ = cmd._collect_margin_findings(Path(pdf_path), tol=tol, limit=limit)
    return findings
