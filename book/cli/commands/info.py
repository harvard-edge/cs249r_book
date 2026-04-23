"""
``binder info`` — Book statistics and figure extraction.

Subcommands:
    stats    — Count figures, tables, equations, listings, text lines, words
    figures  — Extract figure list with labels, captions, and alt-text
               Use --with-pdf to merge LaTeX figure numbers and page numbers
               from a previous PDF build.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FIGURE_DEF = [
    re.compile(r"\{#(fig-[\w-]+)"),
    re.compile(r"#\|\s*label:\s*(fig-[\w-]+)"),
]
_TABLE_DEF = [re.compile(r"\{#(tbl-[\w-]+)")]
_EQUATION_DEF = [re.compile(r"\{#(eq-[\w-]+)")]
_LISTING_DEF = [re.compile(r"\{#(lst-[\w-]+)")]
_SECTION_DEF = [re.compile(r"\{#(sec-[\w-]+)")]

_CODE_FENCE = re.compile(r"^```")
_YAML_FENCE = re.compile(r"^---\s*$")

# Figure extraction patterns — greedy .* anchored to end-of-line so interior
# braces from LaTeX (e.g. $W_{hh}$, \mathcal{F}) don't truncate the match.
_DIV_FIG = re.compile(
    r"^:{3,}\s*\{(.*#fig-.*)\}\s*$", re.MULTILINE
)
_IMG_FIG = re.compile(
    r"!\[((?:[^\[\]]|\[[^\]]*\])*)\]"
    r"\([^)]+\)"
    r"\{(.*#fig-[^\n]*)\}\s*$",
    re.MULTILINE,
)
_ATTR_CAP = re.compile(r'fig-cap\s*=\s*"([^"]*)"')
_ATTR_ALT = re.compile(r'fig-alt\s*=\s*"([^"]*)"')

# Code-block figure pattern: ```{python} with #| cell options
_CODE_BLOCK_FIG = re.compile(
    r"```\{(?:python|r|julia|ojs)\}[^\n]*\n"
    r"((?:#\|[^\n]*\n)+)",
    re.MULTILINE,
)

# LaTeX manifest
_MANIFEST_HEADER = "LATEX FIGURE MANIFEST"
_LATEX_FIG_PAT = re.compile(r"Figure\s+([A-Z\d]+\.\d+)\s*\|\s*Page\s*(\d+)")


class InfoCommand:
    """Native ``binder info`` command group."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, args: List[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder info",
            description="Book statistics and figure extraction",
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=["stats", "figures", "concepts", "headers", "acronyms"],
            help="Subcommand to run",
        )
        parser.add_argument("--path", default=None, help="File or directory")
        parser.add_argument("--vol1", action="store_true", help="Volume I only")
        parser.add_argument("--vol2", action="store_true", help="Volume II only")
        parser.add_argument(
            "--format",
            choices=["text", "markdown", "csv"],
            default="text",
            help="Output format for figures (default: text)",
        )
        parser.add_argument("--output", default=None, help="Write to file")
        parser.add_argument("--json", action="store_true", help="JSON output (stats)")
        parser.add_argument(
            "--by-chapter",
            action="store_true",
            help="Break down stats per chapter",
        )
        parser.add_argument(
            "--with-pdf",
            action="store_true",
            help="Merge LaTeX figure numbers/pages from a previous PDF build",
        )
        parser.add_argument(
            "--manifest",
            default=None,
            help="Path to LaTeX figure manifest (*_figures.txt). Auto-detected if omitted.",
        )

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            self._print_help()
            return True

        root = self._resolve_path(ns.path, ns.vol1, ns.vol2)
        if not root.exists():
            console.print(f"[red]Path not found: {root}[/red]")
            return False

        if ns.subcommand == "stats":
            return self._run_stats(root, ns)
        elif ns.subcommand == "figures":
            return self._run_figures(root, ns)
        elif ns.subcommand == "concepts":
            return self._run_concepts(root, ns)
        elif ns.subcommand == "headers":
            return self._run_headers(root, ns)
        elif ns.subcommand == "acronyms":
            return self._run_acronyms(root, ns)
        return False

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Subcommand", style="cyan", width=14)
        table.add_column("Description", style="white", width=50)
        table.add_row("stats", "Count figures, tables, equations, listings, words, text lines")
        table.add_row("figures", "Extract figure list with labels, captions, and alt-text")
        table.add_row("concepts", "Extract key concepts (bold terms, definitions, headers)")
        table.add_row("headers", "List all section headers with levels")
        table.add_row("acronyms", "Find acronyms in parentheses (e.g., (CNN))")
        console.print(Panel(table, title="binder info <subcommand>", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder info stats --vol1[/cyan]")
        console.print("  [cyan]./binder info stats --by-chapter[/cyan]")
        console.print("  [cyan]./binder info figures --vol1[/cyan]")
        console.print("  [cyan]./binder info figures --vol1 --with-pdf[/cyan]            [dim]# merge LaTeX fig numbers + pages[/dim]")
        console.print("  [cyan]./binder info concepts --vol1[/cyan]")
        console.print("  [cyan]./binder info headers --vol1[/cyan]")
        console.print("  [cyan]./binder info acronyms --vol1[/cyan]")
        console.print()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, path_arg: Optional[str], vol1: bool, vol2: bool) -> Path:
        if path_arg:
            p = Path(path_arg)
            return p if p.is_absolute() else Path.cwd() / p
        base = self.config_manager.book_dir
        if vol1:
            return base / "contents" / "vol1"
        if vol2:
            return base / "contents" / "vol2"
        return base / "contents"

    def _qmd_files(self, root: Path) -> List[Path]:
        if root.is_file():
            return [root] if root.suffix == ".qmd" else []
        return sorted(root.rglob("*.qmd"))

    def _chapter_name(self, path: Path) -> str:
        """Extract a human-readable chapter name from a QMD file."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return path.stem
        # Skip YAML front matter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:]
        m = re.search(r"^#\s+([^{\n]+)", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
        return path.stem.replace("_", " ").title()

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config_manager.book_dir))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _run_stats(self, root: Path, ns: argparse.Namespace) -> bool:
        files = self._qmd_files(root)
        if not files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        # Filter out parts/, frontmatter/ for cleaner stats
        chapter_files = [
            f for f in files
            if "/parts/" not in str(f) and "/frontmatter/" not in str(f)
        ]

        all_stats: List[Dict] = []
        for path in chapter_files:
            stats = self._count_file(path)
            stats["file"] = self._relative(path)
            stats["chapter"] = self._chapter_name(path)
            all_stats.append(stats)

        if ns.json:
            import json
            totals = self._aggregate(all_stats)
            payload = {
                "path": str(root),
                "files": len(all_stats),
                "totals": totals,
            }
            if ns.by_chapter:
                payload["chapters"] = all_stats
            print(json.dumps(payload, indent=2))
            return True

        # Rich table output
        totals = self._aggregate(all_stats)

        if ns.by_chapter:
            self._print_chapter_stats(all_stats, totals)
        else:
            self._print_summary_stats(all_stats, totals, root)

        return True

    def _count_file(self, path: Path) -> Dict:
        """Count elements in a single QMD file."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return self._empty_stats()

        lines = content.splitlines()
        in_code = False
        in_yaml = False
        text_lines = 0
        word_count = 0
        code_blocks = 0
        figures = 0
        tables = 0
        equations = 0
        listings = 0
        sections = 0
        footnotes = 0
        citations = 0

        citation_pat = re.compile(r"@[\w-]+")
        footnote_def_pat = re.compile(r"^\[\^[^\]]+\]:")

        for idx, line in enumerate(lines):
            stripped = line.strip()

            # YAML front matter
            if idx == 0 and _YAML_FENCE.match(stripped):
                in_yaml = True
                continue
            if in_yaml:
                if _YAML_FENCE.match(stripped):
                    in_yaml = False
                continue

            # Code blocks
            if _CODE_FENCE.match(stripped):
                if not in_code:
                    code_blocks += 1
                in_code = not in_code
                continue
            if in_code:
                # Count figure labels inside code blocks
                for pat in _FIGURE_DEF:
                    figures += len(pat.findall(line))
                continue

            # Blank lines don't count as text
            if not stripped:
                continue

            # Skip div fences, HTML comments, raw LaTeX for text counting
            if stripped.startswith(":::") or stripped.startswith("<!--"):
                # But still check for labels in div lines
                for pat in _FIGURE_DEF:
                    figures += len(pat.findall(line))
                for pat in _TABLE_DEF:
                    tables += len(pat.findall(line))
                for pat in _LISTING_DEF:
                    listings += len(pat.findall(line))
                continue

            # Count definitions
            for pat in _FIGURE_DEF:
                figures += len(pat.findall(line))
            for pat in _TABLE_DEF:
                tables += len(pat.findall(line))
            for pat in _EQUATION_DEF:
                equations += len(pat.findall(line))
            for pat in _LISTING_DEF:
                listings += len(pat.findall(line))
            for pat in _SECTION_DEF:
                sections += len(pat.findall(line))

            # Footnote definitions
            if footnote_def_pat.match(stripped):
                footnotes += 1

            # Citations (rough count of unique @cite patterns, excluding cross-refs)
            for cm in citation_pat.finditer(line):
                ref = cm.group(0)[1:]  # strip @
                if not any(ref.startswith(p) for p in ("fig-", "tbl-", "sec-", "eq-", "lst-")):
                    citations += 1

            # Text lines and words (prose lines only)
            # Skip heading attribute lines, pipe table separators, etc.
            if stripped.startswith("#|") or stripped.startswith("%%|"):
                continue
            if stripped.startswith("|") and set(stripped.replace("|", "").strip()) <= {"-", ":", " "}:
                continue  # table separator line

            text_lines += 1
            # Word count: strip markdown formatting, count words
            clean = re.sub(r"`\{python\}[^`]*`", "PYVAL", stripped)
            clean = re.sub(r"\{[^}]+\}", "", clean)  # strip attributes
            clean = re.sub(r"[#*_`~\[\](){}|>]", " ", clean)
            words = [w for w in clean.split() if len(w) > 0 and not w.startswith("\\")]
            word_count += len(words)

        return {
            "figures": figures,
            "tables": tables,
            "equations": equations,
            "listings": listings,
            "sections": sections,
            "footnotes": footnotes,
            "citations": citations,
            "code_blocks": code_blocks,
            "text_lines": text_lines,
            "words": word_count,
        }

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            "figures": 0, "tables": 0, "equations": 0, "listings": 0,
            "sections": 0, "footnotes": 0, "citations": 0,
            "code_blocks": 0, "text_lines": 0, "words": 0,
        }

    @staticmethod
    def _aggregate(stats_list: List[Dict]) -> Dict:
        totals: Dict = {}
        for s in stats_list:
            for k, v in s.items():
                if isinstance(v, int):
                    totals[k] = totals.get(k, 0) + v
        return totals

    def _print_summary_stats(self, all_stats, totals, root) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Count", style="white", justify="right", width=10)

        table.add_row("Chapters", str(len(all_stats)))
        table.add_row("Sections", f"{totals['sections']:,}")
        table.add_row("Figures", f"{totals['figures']:,}")
        table.add_row("Tables", f"{totals['tables']:,}")
        table.add_row("Equations", f"{totals['equations']:,}")
        table.add_row("Listings", f"{totals['listings']:,}")
        table.add_row("Footnotes", f"{totals['footnotes']:,}")
        table.add_row("Citations", f"{totals['citations']:,}")
        table.add_row("Code blocks", f"{totals['code_blocks']:,}")
        table.add_row("Text lines", f"{totals['text_lines']:,}")
        table.add_row("Words (approx)", f"{totals['words']:,}")

        scope = str(root)
        try:
            scope = str(root.relative_to(self.config_manager.book_dir))
        except ValueError:
            pass
        console.print(Panel(table, title=f"Book Statistics — {scope}", border_style="cyan"))

    def _print_chapter_stats(self, all_stats, totals) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Chapter", style="white", width=32, no_wrap=True)
        table.add_column("Fig", justify="right", width=5)
        table.add_column("Tbl", justify="right", width=5)
        table.add_column("Eq", justify="right", width=5)
        table.add_column("Lst", justify="right", width=5)
        table.add_column("Sec", justify="right", width=5)
        table.add_column("Fn", justify="right", width=5)
        table.add_column("Cite", justify="right", width=6)
        table.add_column("Words", justify="right", width=8)

        for s in all_stats:
            name = s["chapter"]
            if len(name) > 30:
                name = name[:28] + "…"
            table.add_row(
                name,
                str(s["figures"]),
                str(s["tables"]),
                str(s["equations"]),
                str(s["listings"]),
                str(s["sections"]),
                str(s["footnotes"]),
                str(s["citations"]),
                f"{s['words']:,}",
            )

        # Totals row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{totals['figures']}[/bold]",
            f"[bold]{totals['tables']}[/bold]",
            f"[bold]{totals['equations']}[/bold]",
            f"[bold]{totals['listings']}[/bold]",
            f"[bold]{totals['sections']}[/bold]",
            f"[bold]{totals['footnotes']}[/bold]",
            f"[bold]{totals['citations']}[/bold]",
            f"[bold]{totals['words']:,}[/bold]",
        )

        console.print(Panel(table, title="Book Statistics — By Chapter", border_style="cyan"))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _run_figures(self, root: Path, ns: argparse.Namespace) -> bool:
        with_pdf = getattr(ns, "with_pdf", False)

        # Determine chapter file list
        if with_pdf:
            # Use config-ordered list so sequential merge with LaTeX works
            vol = "vol1" if ns.vol1 else ("vol2" if ns.vol2 else None)
            chapter_files = self._config_ordered_chapters(vol)
            if not chapter_files:
                console.print("[yellow]Could not read chapter order from config. Falling back to directory scan.[/yellow]")
                chapter_files = self._chapter_files_from_root(root)
        else:
            chapter_files = self._chapter_files_from_root(root)

        if not chapter_files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        # Extract QMD figures
        all_figures: List[Dict] = []
        for path in chapter_files:
            chapter = self._chapter_name(path)
            figs = self._extract_figures(path)
            for i, fig in enumerate(figs, 1):
                fig["chapter"] = chapter
                fig["file"] = self._relative(path)
                fig["seq"] = i  # per-chapter sequential number
            all_figures.extend(figs)

        if not all_figures:
            console.print("[yellow]No figures found.[/yellow]")
            return True

        # Merge with LaTeX manifest if requested
        if with_pdf:
            vol = "vol1" if ns.vol1 else ("vol2" if ns.vol2 else None)
            manifest_path = self._find_latex_manifest(ns.manifest, vol)
            if manifest_path:
                latex_figs = self._parse_latex_manifest(manifest_path)
                console.print(
                    f"[dim]LaTeX manifest: {manifest_path.name} "
                    f"({len(latex_figs)} figures)[/dim]"
                )
                if len(latex_figs) != len(all_figures):
                    console.print(
                        f"[yellow]Warning: LaTeX ({len(latex_figs)}) and QMD "
                        f"({len(all_figures)}) figure counts differ. "
                        f"Manifest may be stale — rebuild PDF to fix.[/yellow]"
                    )
                # Merge by sequential position
                for i, fig in enumerate(all_figures):
                    if i < len(latex_figs):
                        fig["fig_number"] = latex_figs[i]["number"]
                        fig["page"] = latex_figs[i]["page"]
                    else:
                        fig["fig_number"] = "?"
                        fig["page"] = "?"
            else:
                console.print(
                    "[yellow]No LaTeX manifest found. Build PDF first, "
                    "or pass --manifest PATH.[/yellow]"
                )
                for fig in all_figures:
                    fig["fig_number"] = "?"
                    fig["page"] = "?"
        else:
            # No PDF data — assign per-chapter sequential numbers
            for fig in all_figures:
                fig["fig_number"] = ""
                fig["page"] = ""

        output = self._format_figures(all_figures, ns.format, with_pdf)

        if ns.output:
            Path(ns.output).write_text(output, encoding="utf-8")
            console.print(f"[green]Wrote {len(all_figures)} figures to {ns.output}[/green]")
        else:
            if ns.format == "text":
                self._print_figures_rich(all_figures, with_pdf)
            else:
                print(output)

        return True

    def _chapter_files_from_root(self, root: Path) -> List[Path]:
        """Get chapter QMD files from a directory, filtering out parts/."""
        files = self._qmd_files(root)
        return [f for f in files if "/parts/" not in str(f)]

    # ------------------------------------------------------------------
    # Config reading
    # ------------------------------------------------------------------

    def _pdf_config_paths(self, vol: Optional[str] = None) -> List[Path]:
        """Return candidate PDF config YAML paths in priority order."""
        quarto_dir = self.config_manager.book_dir
        vol_str = vol or "vol1"
        return [
            quarto_dir / "_quarto.yml",
            quarto_dir / f"config/_quarto-pdf-{vol_str}.yml",
            quarto_dir / f"config/_quarto-pdf-{vol_str}-copyedit.yml",
        ]

    def _read_pdf_config(self, vol: Optional[str] = None) -> Tuple[Optional[Path], Optional[dict], str]:
        """Read the first valid PDF config for *vol*.

        Returns (config_path, parsed_yaml, raw_text).  Any element may
        be ``None`` / empty if no config was found.
        """
        import yaml

        for config_path in self._pdf_config_paths(vol):
            if not config_path.exists():
                continue
            resolved = config_path.resolve() if config_path.is_symlink() else config_path
            try:
                raw = resolved.read_text(encoding="utf-8")
                parsed = yaml.safe_load(raw)
            except Exception:
                continue
            if parsed and "book" in parsed:
                return resolved, parsed, raw
        return None, None, ""

    def _output_dirs_from_configs(self, vol: Optional[str] = None) -> List[Path]:
        """Return all ``output-dir`` values declared in the PDF configs for *vol*.

        This is the authoritative way to locate build artifacts — the
        YAML configs define exactly where Quarto writes its output.
        """
        import yaml

        quarto_dir = self.config_manager.book_dir
        dirs: List[Path] = []

        for config_path in self._pdf_config_paths(vol):
            if not config_path.exists():
                continue
            resolved = config_path.resolve() if config_path.is_symlink() else config_path
            try:
                parsed = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not parsed:
                continue
            output_dir = (parsed.get("project") or {}).get("output-dir")
            if output_dir:
                full = quarto_dir / output_dir
                if full not in dirs:
                    dirs.append(full)

        return dirs

    def _config_ordered_chapters(self, vol: Optional[str] = None) -> List[Path]:
        """Read chapter order from Quarto YAML config (same order as PDF build).

        This is critical for --with-pdf because the LaTeX manifest is
        sequential — Figure N in the manifest corresponds to the Nth
        figure extracted from the QMD files in config order.
        """
        quarto_dir = self.config_manager.book_dir
        vol_str = vol or "vol1"

        _, _, raw = self._read_pdf_config(vol)
        if not raw:
            return []

        qmd_files: List[Path] = []

        # Read both commented and uncommented chapter entries (full intended order)
        # Pattern: lines like "    - contents/vol1/chapter/chapter.qmd"
        # or "    # - contents/vol1/chapter/chapter.qmd"
        comment_pat = re.compile(
            rf"^\s*#?\s*-\s*(contents/{vol_str}/[^\s#]+\.qmd)\s*$",
            re.MULTILINE,
        )
        for m in comment_pat.finditer(raw):
            rel = m.group(1)
            # Skip parts, frontmatter, shelved
            if "/parts/" in rel or "/frontmatter/" in rel or "_shelved" in rel:
                continue
            full = quarto_dir / rel
            if full.exists():
                qmd_files.append(full)

        return qmd_files

    # ------------------------------------------------------------------
    # LaTeX manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _is_latex_manifest(path: Path) -> bool:
        try:
            return _MANIFEST_HEADER in path.read_text(encoding="utf-8")[:200]
        except Exception:
            return False

    def _find_latex_manifest(
        self, explicit_path: Optional[str], vol: Optional[str]
    ) -> Optional[Path]:
        """Locate the LaTeX figure manifest (*_figures.txt).

        Search order:
          1. Explicit --manifest path
          2. ``output-dir`` directories declared in the PDF YAML configs
          3. Quarto root directory (fresh from LaTeX, not yet moved)
        """
        if explicit_path:
            p = Path(explicit_path)
            if not p.is_absolute():
                p = Path.cwd() / p
            return p if p.exists() else None

        quarto_dir = self.config_manager.book_dir
        candidates: List[Path] = []

        # Search the output directories declared in the YAML configs
        for output_dir in self._output_dirs_from_configs(vol):
            if output_dir.exists():
                for f in output_dir.glob("*_figures.txt"):
                    if self._is_latex_manifest(f):
                        candidates.append(f)

        # Fallback: quarto root (LaTeX writes here during compilation,
        # before the post-render script moves it into the build dir)
        for f in quarto_dir.glob("*_figures.txt"):
            if self._is_latex_manifest(f):
                candidates.append(f)

        if not candidates:
            return None

        # Most recently modified wins
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    @staticmethod
    def _parse_latex_manifest(path: Path) -> List[Dict]:
        """Parse a LaTeX figure manifest into [{number, page}, ...]."""
        figures: List[Dict] = []
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return figures
        for m in _LATEX_FIG_PAT.finditer(content):
            figures.append({"number": m.group(1), "page": m.group(2)})
        return figures

    # ------------------------------------------------------------------
    # Figure extraction from QMD
    # ------------------------------------------------------------------

    def _extract_figures(self, path: Path) -> List[Dict]:
        """Extract all figures from a QMD file, deduplicating by label."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return []

        raw: List[Dict] = []

        # 1. Fenced divs: ::: {#fig-id fig-cap="..." fig-alt="..."}
        for m in _DIV_FIG.finditer(content):
            attrs = m.group(1)
            label_m = re.search(r"#(fig-[\w-]+)", attrs)
            if not label_m:
                continue
            fig_id = label_m.group(1)
            cap_m = _ATTR_CAP.search(attrs)
            alt_m = _ATTR_ALT.search(attrs)
            caption = self._unescape(cap_m.group(1)) if cap_m else ""
            alt_text = self._unescape(alt_m.group(1)) if alt_m else ""
            raw.append({
                "id": fig_id,
                "label": self._extract_title(caption),
                "caption": caption,
                "alt_text": alt_text,
                "source": "div",
                "pos": m.start(),
            })

        # 2. Markdown images: ![cap](path){#fig-id fig-alt="..."}
        for m in _IMG_FIG.finditer(content):
            attrs = m.group(2)
            label_m = re.search(r"#(fig-[\w-]+)", attrs)
            if not label_m:
                continue
            fig_id = label_m.group(1)
            caption = m.group(1).strip()
            cap_m = _ATTR_CAP.search(attrs)
            alt_m = _ATTR_ALT.search(attrs)
            if cap_m:
                caption = self._unescape(cap_m.group(1))
            alt_text = self._unescape(alt_m.group(1)) if alt_m else ""
            raw.append({
                "id": fig_id,
                "label": self._extract_title(caption),
                "caption": caption,
                "alt_text": alt_text,
                "source": "image",
                "pos": m.start(),
            })

        # 3. Code-cell figures: ```{python} with #| label: fig-...
        for m in _CODE_BLOCK_FIG.finditer(content):
            cell_opts_text = m.group(1)
            lbl_m = re.search(r"#\|\s*label:\s*(fig-[\w-]+)", cell_opts_text)
            if not lbl_m:
                continue
            fig_id = lbl_m.group(1)
            cap_m = re.search(r'#\|\s*fig-cap:\s*"([^"]*)"', cell_opts_text)
            alt_m = re.search(r'#\|\s*fig-alt:\s*"([^"]*)"', cell_opts_text)
            caption = cap_m.group(1) if cap_m else ""
            alt_text = alt_m.group(1) if alt_m else ""
            raw.append({
                "id": fig_id,
                "label": self._extract_title(caption),
                "caption": caption,
                "alt_text": alt_text,
                "source": "code-cell",
                "pos": m.start(),
            })

        # Sort by position in file, then deduplicate by label
        raw.sort(key=lambda x: x["pos"])
        seen: Dict[str, Dict] = {}
        for fig in raw:
            fid = fig["id"]
            if fid in seen:
                # Keep whichever has a richer caption
                if not seen[fid]["caption"] and fig["caption"]:
                    seen[fid] = fig
            else:
                seen[fid] = fig

        return [
            {k: v for k, v in fig.items() if k != "pos"}
            for fig in sorted(seen.values(), key=lambda x: x["pos"])
        ]

    @staticmethod
    def _unescape(s: str) -> str:
        return s.replace('\\"', '"').replace("\\'", "'")

    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        return s

    @staticmethod
    def _extract_title(caption: str) -> str:
        """Extract the bold title from a caption."""
        if not caption:
            return ""
        m = re.match(r"\*\*([^*]+)\*\*", caption)
        if m:
            return m.group(1).rstrip(":.").strip()
        colon = caption.find(":")
        period = caption.find(".")
        if colon > 0 and (period < 0 or colon < period):
            return caption[:colon].strip()
        if period > 0:
            return caption[:period].strip()
        return caption[:60].strip()

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _fig_display_number(self, fig: Dict, with_pdf: bool) -> str:
        """Return the display number for a figure."""
        if with_pdf and fig.get("fig_number"):
            return fig["fig_number"]
        return str(fig.get("seq", ""))

    def _format_figures(self, figures: List[Dict], fmt: str, with_pdf: bool = False) -> str:
        if fmt == "csv":
            header = "chapter,fig_number,page,label,id,caption,alt_text,source,file"
            lines = [header]
            for f in figures:
                cap = f["caption"].replace('"', '""')
                alt = f["alt_text"].replace('"', '""')
                lbl = f["label"].replace('"', '""')
                fig_num = f.get("fig_number", "")
                page = f.get("page", "")
                lines.append(
                    f'"{f["chapter"]}",{fig_num},{page},"{lbl}",{f["id"]},"{cap}","{alt}",{f["source"]},{f["file"]}'
                )
            return "\n".join(lines) + "\n"

        if fmt == "markdown":
            title = "# Figure List"
            if with_pdf:
                title += " (with PDF numbers)"
            lines = [f"{title}\n"]
            current_chapter = ""
            for f in figures:
                if f["chapter"] != current_chapter:
                    current_chapter = f["chapter"]
                    lines.append(f"\n## {current_chapter}\n")
                num = self._fig_display_number(f, with_pdf)
                page = f.get("page", "")
                header = f"**Figure {num}**"
                if with_pdf and page:
                    header += f" (Page {page})"
                header += f" (`{f['id']}`)"
                lines.append(header)
                if f["label"]:
                    lines.append(f"  **Title**: {f['label']}")
                if f["caption"]:
                    lines.append(f"  **Caption**: {f['caption']}")
                if f["alt_text"]:
                    lines.append(f"  **Alt-text**: {f['alt_text']}")
                lines.append(f"  *Source*: {f['source']} | *File*: {f['file']}")
                lines.append("")
            # Summary
            total = len(figures)
            with_cap = sum(1 for f in figures if f["caption"])
            with_alt = sum(1 for f in figures if f["alt_text"])
            lines.append(f"---\n**Total**: {total} figures | Captions: {with_cap}/{total} | Alt-text: {with_alt}/{total}")
            return "\n".join(lines)

        # Plain text
        lines = []
        current_chapter = ""
        for f in figures:
            if f["chapter"] != current_chapter:
                current_chapter = f["chapter"]
                lines.append(f"\n{'='*70}")
                lines.append(f"  {current_chapter}")
                lines.append(f"{'='*70}")
            num = self._fig_display_number(f, with_pdf)
            page = f.get("page", "")
            if with_pdf and page:
                lines.append(f"  Figure {num:>6s}  (Page {page:>4s})  {f['id']}")
            else:
                lines.append(f"  Fig {num:>3s}  {f['id']}")
            if f["label"]:
                lines.append(f"          Title: {f['label']}")
            has_cap = "YES" if f["caption"] else "MISSING"
            has_alt = "YES" if f["alt_text"] else "MISSING"
            lines.append(f"          Caption: {has_cap}  |  Alt-text: {has_alt}")
        lines.append("")
        total = len(figures)
        with_cap = sum(1 for f in figures if f["caption"])
        with_alt = sum(1 for f in figures if f["alt_text"])
        lines.append(f"Total: {total} figures  |  Captions: {with_cap}/{total}  |  Alt-text: {with_alt}/{total}")
        return "\n".join(lines) + "\n"

    def _print_figures_rich(self, figures: List[Dict], with_pdf: bool = False) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        if with_pdf:
            table.add_column("Fig #", style="bold", width=8, justify="right")
            table.add_column("Page", style="dim", width=5, justify="right")
        else:
            table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Label", style="cyan", width=30, no_wrap=True)
        table.add_column("Title", style="white", width=30, no_wrap=True)
        table.add_column("Cap", width=4, justify="center")
        table.add_column("Alt", width=4, justify="center")
        table.add_column("Src", style="dim", width=6)

        current_chapter = ""
        for f in figures:
            if f["chapter"] != current_chapter:
                current_chapter = f["chapter"]
                name = current_chapter[:30] if len(current_chapter) <= 30 else current_chapter[:28] + "…"
                if with_pdf:
                    table.add_row("", "", f"[bold yellow]{name}[/bold yellow]", "", "", "", "")
                else:
                    table.add_row("", f"[bold yellow]{name}[/bold yellow]", "", "", "", "")

            title = f["label"][:28] + "…" if len(f["label"]) > 30 else f["label"]
            cap = "[green]✓[/green]" if f["caption"] else "[red]✗[/red]"
            alt = "[green]✓[/green]" if f["alt_text"] else "[red]✗[/red]"
            num = self._fig_display_number(f, with_pdf)

            if with_pdf:
                page = f.get("page", "")
                table.add_row(num, page, f["id"], title, cap, alt, f["source"])
            else:
                table.add_row(num, f["id"], title, cap, alt, f["source"])

        total = len(figures)
        with_cap = sum(1 for f in figures if f["caption"])
        with_alt = sum(1 for f in figures if f["alt_text"])

        panel_title = "Figure List"
        if with_pdf:
            panel_title += " (with PDF numbers)"
        console.print(Panel(table, title=panel_title, border_style="cyan"))
        console.print(
            f"[bold]Total[/bold]: {total} figures  |  "
            f"Captions: {with_cap}/{total}  |  "
            f"Alt-text: {with_alt}/{total}"
        )

    # ------------------------------------------------------------------
    # Concepts  (ported from extract_concepts.py)
    # ------------------------------------------------------------------

    def _run_concepts(self, root: Path, ns) -> bool:
        """Extract key concepts from QMD files."""
        files = self._qmd_files(root)
        chapter_files = [f for f in files if "/parts/" not in str(f) and "/frontmatter/" not in str(f)]

        if not chapter_files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        bold_pat = re.compile(r"\*\*([^*]+)\*\*")
        fn_def_pat = re.compile(r"\[\^fn-([^\]]+)\]:\s*(.+?)(?=\n\n|\[\^|\Z)", re.DOTALL)
        def_patterns = [
            re.compile(r"(\w[\w\s]+?)\s+is defined as", re.IGNORECASE),
            re.compile(r"(\w[\w\s]+?)\s+refers to", re.IGNORECASE),
            re.compile(r"We define\s+(\w[\w\s]+?)\s+as", re.IGNORECASE),
        ]
        heading_pat = re.compile(r"^(#{1,6})\s+(.*)")

        for path in chapter_files:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = content.splitlines()
            chapter = self._chapter_name(path)

            # Extract H2 topics
            h2_topics = []
            for line in lines:
                m = heading_pat.match(line)
                if m and len(m.group(1)) == 2:
                    text = re.sub(r"\{#[^}]+\}", "", m.group(2)).strip()
                    if not text.startswith("Purpose"):
                        h2_topics.append(text)

            # Bold terms
            bold_terms = set()
            for m in bold_pat.finditer(content):
                term = m.group(1).strip()
                if len(term) > 2 and not term.startswith("Note"):
                    bold_terms.add(term)

            # Definitions
            definitions = set()
            for pat in def_patterns:
                for m in pat.finditer(content):
                    term = m.group(1).strip()
                    if len(term) < 50:
                        definitions.add(term)

            # Footnotes
            footnotes = []
            for m in fn_def_pat.finditer(content):
                footnotes.append(m.group(1))

            # Print
            name = chapter[:40] if len(chapter) <= 40 else chapter[:38] + "…"
            console.print(f"\n[bold cyan]{name}[/bold cyan]  [dim]({self._relative(path)})[/dim]")

            if h2_topics:
                console.print(f"  [yellow]Topics:[/yellow] {', '.join(h2_topics[:8])}")
            if bold_terms:
                terms = sorted(bold_terms)[:12]
                console.print(f"  [yellow]Key terms:[/yellow] {', '.join(terms)}")
            if definitions:
                defs = sorted(definitions)[:6]
                console.print(f"  [yellow]Defines:[/yellow] {', '.join(defs)}")
            if footnotes:
                console.print(f"  [yellow]Footnotes:[/yellow] {len(footnotes)}")

        console.print(f"\n[bold]Scanned {len(chapter_files)} files.[/bold]")
        return True

    # ------------------------------------------------------------------
    # Headers  (ported from extract_headers.py)
    # ------------------------------------------------------------------

    def _run_headers(self, root: Path, ns) -> bool:
        """List all section headers with levels."""
        files = self._qmd_files(root)
        chapter_files = [f for f in files if "/parts/" not in str(f)]

        if not chapter_files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        heading_pat = re.compile(r"^(#{1,6})\s+(.*)")

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("File", style="dim", width=30, no_wrap=True)
        table.add_column("Lvl", style="cyan", width=5, justify="center")
        table.add_column("Header", style="white", width=50)

        total = 0
        for path in chapter_files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            rel = self._relative(path)
            if len(rel) > 28:
                rel = "…" + rel[-27:]
            for line in lines:
                m = heading_pat.match(line)
                if m:
                    level = m.group(1)
                    text = m.group(2).strip()
                    # Indent based on level for visual hierarchy
                    indent = "  " * (len(level) - 1)
                    table.add_row(rel, level, f"{indent}{text}")
                    total += 1
                    rel = ""  # Only show filename on first header

        console.print(Panel(table, title="Section Headers", border_style="cyan"))
        console.print(f"[bold]Total:[/bold] {total} headers across {len(chapter_files)} files")
        return True

    # ------------------------------------------------------------------
    # Acronyms  (ported from find_acronyms.py)
    # ------------------------------------------------------------------

    def _run_acronyms(self, root: Path, ns) -> bool:
        """Find acronyms in parentheses (e.g., (CNN), (GPU))."""
        from collections import Counter

        files = self._qmd_files(root)
        if not files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        acronym_pat = re.compile(r"\(([A-Z]{2,}s?)\)")
        acronym_counts: Dict[str, int] = Counter()
        acronym_files: Dict[str, List[str]] = defaultdict(list)

        for path in files:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            found = set()
            for m in acronym_pat.finditer(content):
                acr = m.group(1)
                acronym_counts[acr] += 1
                found.add(acr)
            for acr in found:
                acronym_files[acr].append(self._relative(path))

        if not acronym_counts:
            console.print("[yellow]No acronyms found.[/yellow]")
            return True

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Acronym", style="cyan", width=12)
        table.add_column("Count", style="white", width=7, justify="right")
        table.add_column("Files", style="dim", width=50)

        for acr, count in sorted(acronym_counts.items(), key=lambda x: -x[1]):
            file_list = acronym_files[acr]
            files_str = ", ".join(f.split("/")[-1].replace(".qmd", "") for f in file_list[:5])
            if len(file_list) > 5:
                files_str += f" +{len(file_list) - 5} more"
            table.add_row(acr, str(count), files_str)

        console.print(Panel(table, title="Acronyms Found", border_style="cyan"))
        console.print(f"[bold]Total:[/bold] {len(acronym_counts)} unique acronyms, {sum(acronym_counts.values())} occurrences")
        return True
