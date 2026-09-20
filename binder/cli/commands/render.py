"""
``binder render`` — Render generated figures to a browsable gallery.

Subcommands:
    plots    — Render all matplotlib/Python figures from QMD files to PNG.
               Outputs to ``_output/plots/<chapter>/<fig-label>.png``.

Future subcommands:
    diagrams — Render TikZ diagrams (requires lualatex)
    all      — Render everything
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional  # noqa: UP035

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```\{python\}\s*\n(.*?)```", re.DOTALL)
_FIG_LABEL = re.compile(r"#\|\s*label:\s*(fig-[\w-]+)")


def _extract_python_figures(qmd_path: Path) -> List[Dict]:
    """Extract Python code blocks that have a ``fig-*`` label."""
    content = qmd_path.read_text(encoding="utf-8")
    figures: List[Dict] = []

    for match in _CODE_BLOCK.finditer(content):
        block = match.group(1)
        label_match = _FIG_LABEL.search(block)
        if not label_match:
            continue

        label = label_match.group(1)

        # Strip #| directives to get executable Python
        lines = block.split("\n")
        code_lines = [ln for ln in lines if not ln.strip().startswith("#|")]
        code = "\n".join(code_lines).strip()

        figures.append({"label": label, "code": code})

    return figures


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_one(code: str, output_path: str) -> Optional[str]:
    """Execute a figure's Python code and save to PNG.

    Returns None on success, or an error message string on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Replace any direct show call with a save-and-close sequence.
    modified = re.sub(
        r"plt\.show\(\)",
        f"plt.savefig('{output_path}', dpi=150, bbox_inches='tight')\nplt.close('all')",
        code,
    )

    # If code never calls plt.show(), append savefig
    if "plt.show()" not in code and "savefig" not in code:
        modified += (
            f"\nplt.savefig('{output_path}', dpi=150, bbox_inches='tight')"
            f"\nplt.close('all')"
        )

    try:
        exec(modified, {"__name__": "__main__"})  # noqa: S102
        return None
    except Exception as e:
        return str(e)
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class RenderCommand:
    """Handles ``binder render`` operations."""

    def __init__(self, config_manager, chapter_discovery):
        """Store shared managers and the book directory used for scanning and output."""
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.book_dir = config_manager.book_dir

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, args: List[str]) -> bool:
        """Parse ``binder render`` arguments and dispatch; only ``plots`` is implemented.

        With no subcommand (or ``help``) prints help and returns True. Returns
        False on an argparse error (True for ``-h``/``--help``).
        """
        if args == ["help"]:
            self._print_help()
            return True

        parser = argparse.ArgumentParser(
            prog="binder render",
            description="Render generated figures to a browsable gallery",
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=["plots", "figures", "diagrams"],
            help="What to render (plots, figures, diagrams)",
        )
        parser.add_argument("chapters", nargs="?", default=None,
                            help="Chapter name(s), comma-separated")
        parser.add_argument("--vol1", action="store_true", help="Volume I only")
        parser.add_argument("--vol2", action="store_true", help="Volume II only")
        parser.add_argument("--vol3", action="store_true", help="Volume III only")
        parser.add_argument("--vol4", action="store_true", help="Volume IV only")
        parser.add_argument("--chapter", type=str, default="", help="Filter chapter slug(s)")
        parser.add_argument("--type", type=str, default="all", choices=["all", "tikz", "margin", "images", "cover"], help="Filter figure type")
        parser.add_argument("--limit", type=int, default=0, help="Limit number of figures (0 = all)")
        parser.add_argument("--no-render", action="store_true", help="Skip Quarto compile")
        parser.add_argument("--no-contact-sheets", action="store_true", help="Skip contact sheets")
        parser.add_argument("--contact-cols", type=int, default=3, help="Grid columns (default: 3)")
        parser.add_argument("--contact-rows", type=int, default=4, help="Grid rows (default: 4)")
        parser.add_argument("--dpi", type=int, default=110, help="Rasterization DPI (default: 110)")

        try:
            ns, unknown = parser.parse_known_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            self._print_help()
            return True

        if ns.subcommand == "plots":
            return self._render_plots(ns)

        if ns.subcommand in ("figures", "diagrams"):
            return self._render_figures(ns, unknown)

        return False

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        """Print the subcommand table and usage examples."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Subcommand", style="cyan", width=14)
        table.add_column("Description", style="white", width=55)
        table.add_row("plots", "Render matplotlib/Python figures to PNG gallery")
        table.add_row("figures", "Render figure-only PDF and PNG contact sheets")
        console.print(Panel(table, title="binder render <subcommand>", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder/binder render plots[/cyan]                    [dim]# all chapters, both volumes[/dim]")
        console.print("  [cyan]./binder/binder render plots --vol1[/cyan]             [dim]# Volume I only[/dim]")
        console.print("  [cyan]./binder/binder render figures --vol2[/cyan]           [dim]# Volume II figures PDF + contact sheets[/dim]")
        console.print("  [cyan]./binder/binder render figures --vol2 --type tikz[/cyan] [dim]# Volume II TikZ figures only[/dim]")
        console.print()

    def _render_figures(self, ns: argparse.Namespace, unknown: List[str]) -> bool:
        """Render figure contact sheets (TikZ, margin, images)."""
        import sys
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scripts.generate_figure_contact_sheet import FigureContactSheetBuilder

        vol = "vol2" if ns.vol2 else ("vol1" if ns.vol1 else ("vol3" if ns.vol3 else ("vol4" if ns.vol4 else None)))
        if not vol:
            console.print("[red]Please specify a volume: --vol1, --vol2, --vol3, or --vol4[/red]")
            return False

        # Parse chapter filter
        chapter_filter = []
        if ns.chapter:
            chapter_filter = [c.strip() for c in ns.chapter.split(",") if c.strip()]
        elif ns.chapters:
            chapter_filter = [c.strip() for c in ns.chapters.split(",") if c.strip()]

        fig_type = "tikz" if ns.subcommand == "diagrams" else ns.type

        builder = FigureContactSheetBuilder(
            volume=vol,
            chapter_filter=chapter_filter,
            figure_type=fig_type,
            limit=ns.limit,
            dpi=max(36, ns.dpi),
            cols=max(1, ns.contact_cols),
            rows=max(1, ns.contact_rows),
        )
        entries = builder.extract_figures()
        if not entries:
            console.print(f"[yellow]No matching figures found for {vol}.[/yellow]")
            return False
        builder.export_metadata(entries)
        qmd_path = builder.generate_qmd(entries)
        console.print(f"[bold blue]Extracted[/bold blue] {len(entries)} figure{'s' if len(entries) != 1 else ''}.")
        console.print(f"[dim]Audit source:[/dim] {qmd_path}")
        if not ns.no_render:
            pdf_path = builder.render_pdf(qmd_path)
            if pdf_path and not ns.no_contact_sheets:
                sheets = builder.make_contact_sheets(pdf_path)
                if sheets:
                    console.print("[dim]Contact sheets:[/dim]")
                    for sheet in sheets:
                        console.print(f"  {sheet}")
        return True

    # ------------------------------------------------------------------
    # Resolve QMD files
    # ------------------------------------------------------------------

    def _resolve_qmd_files(self, ns: argparse.Namespace) -> List[Path]:
        """Resolve which QMD files to scan based on CLI arguments."""
        contents_dir = self.book_dir

        # Specific chapters requested
        if ns.chapters:
            chapter_names = [c.strip() for c in ns.chapters.split(",")]
            files: List[Path] = []
            for name in chapter_names:
                found = self.chapter_discovery.find_chapter_file(name)
                if found:
                    files.append(found)
                else:
                    # Direct glob fallback
                    matches = list(contents_dir.rglob(f"**/{name}/{name}.qmd"))
                    if matches:
                        files.append(matches[0])
                    else:
                        console.print(f"[yellow]Chapter not found: {name}[/yellow]")
            return files

        # Volume filter
        if ns.vol1:
            root = contents_dir / "vol1"
        elif ns.vol2:
            root = contents_dir / "vol2"
        else:
            root = contents_dir

        if not root.exists():
            console.print(f"[red]Path not found: {root}[/red]")
            return []

        # All QMD files, excluding parts/ and frontmatter/
        return sorted(
            f for f in root.rglob("*.qmd")
            if "/parts/" not in str(f) and "/frontmatter/" not in str(f)
        )

    # ------------------------------------------------------------------
    # Render plots
    # ------------------------------------------------------------------

    def _render_plots(self, ns: argparse.Namespace) -> bool:
        """Execute every ``fig-`` labeled Python cell in-process and save each figure as PNG.

        Output goes to ``<book_dir>/_output/plots/<chapter>/<fig-label>.png``.
        The working directory is switched to the book directory during rendering
        and ``.`` is prepended to ``sys.path``. Prints a per-chapter summary,
        notes TikZ and static figures that need a full build, and opens the
        output folder on macOS. Returns False when no QMD files are found or any
        figure fails to render.
        """
        qmd_files = self._resolve_qmd_files(ns)
        if not qmd_files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        output_base = self.book_dir / "_output" / "plots"

        # Ensure we're in the quarto directory so mlsys imports work
        original_cwd = os.getcwd()
        os.chdir(str(self.book_dir))
        sys.path.insert(0, ".")

        total = 0
        success = 0
        failed = 0
        skipped_chapters = 0
        chapter_results: List[Dict] = []

        # Count non-matplotlib figures for the summary
        tikz_count = 0
        static_count = 0

        console.print()
        console.print("[bold cyan]Rendering matplotlib plots...[/bold cyan]")
        console.print()

        for qmd_path in qmd_files:
            chapter_name = qmd_path.stem
            figures = _extract_python_figures(qmd_path)

            # Count TikZ and static images for summary
            try:
                content = qmd_path.read_text(encoding="utf-8")
                tikz_count += len(re.findall(r"```\{\.tikz\}", content))
                static_count += len(re.findall(
                    r"!\[.*?\]\(.*?\)\{#fig-", content
                ))
            except Exception:
                pass

            if not figures:
                skipped_chapters += 1
                continue

            out_dir = output_base / chapter_name
            out_dir.mkdir(parents=True, exist_ok=True)

            ch_ok = 0
            ch_fail = 0

            for fig in figures:
                total += 1
                out_file = out_dir / f"{fig['label']}.png"

                err = _render_one(fig["code"], str(out_file))
                if err is None:
                    success += 1
                    ch_ok += 1
                else:
                    failed += 1
                    ch_fail += 1
                    console.print(
                        f"  [red]FAIL[/red] {chapter_name}/{fig['label']}: {err}"
                    )

            chapter_results.append({
                "chapter": chapter_name,
                "ok": ch_ok,
                "fail": ch_fail,
                "total": ch_ok + ch_fail,
            })

        # Restore working directory
        os.chdir(original_cwd)

        # --- Summary ---
        console.print()

        if not chapter_results:
            console.print("[yellow]No matplotlib plots found in the selected files.[/yellow]")
            return True

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Chapter", style="white", width=30)
        table.add_column("Plots", justify="right", width=8)
        table.add_column("Status", width=20)

        for r in chapter_results:
            if r["fail"] == 0:
                status = f"[green]{r['ok']} OK[/green]"
            else:
                status = f"[green]{r['ok']} OK[/green], [red]{r['fail']} FAILED[/red]"
            table.add_row(r["chapter"], str(r["total"]), status)

        # Totals row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total}[/bold]",
            f"[bold green]{success} OK[/bold green]"
            + (f", [bold red]{failed} FAILED[/bold red]" if failed else ""),
        )

        console.print(Panel(table, title="Rendered Plots", border_style="cyan"))

        # Note about other figure types
        other = tikz_count + static_count
        if other > 0:
            parts = []
            if tikz_count:
                parts.append(f"{tikz_count} TikZ diagram{'s' if tikz_count != 1 else ''}")
            if static_count:
                parts.append(f"{static_count} static image{'s' if static_count != 1 else ''}")
            console.print(
                f"[dim]{' and '.join(parts)} require "
                f"[cyan]./binder/binder build[/cyan] to render.[/dim]"
            )

        console.print(f"\n[bold]Output:[/bold] {output_base}/")

        # Open folder (default on macOS since the whole point is visual review)
        if platform.system() == "Darwin":
            try:
                subprocess.run(["open", str(output_base)], check=False)
            except Exception:
                pass

        return failed == 0
