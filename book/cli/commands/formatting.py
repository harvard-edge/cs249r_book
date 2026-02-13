"""
Format commands for MLSysBook CLI.

Auto-formatters for QMD content: blank lines, Python code blocks,
list spacing, div spacing, and table formatting.

Usage:
    binder format blanks   — Collapse extra blank lines
    binder format python   — Format Python code blocks (Black, 70 chars)
    binder format lists    — Fix bullet list spacing
    binder format divs     — Fix div/callout spacing
    binder format tables   — Prettify grid tables
    binder format all      — Run all formatters
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Paths to legacy scripts (used for complex formatters not yet natively ported)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "tools" / "scripts"
_SCRIPT_PATHS = {
    "python": _SCRIPTS_DIR / "content" / "format_python_in_qmd.py",
    "tables": _SCRIPTS_DIR / "content" / "format_tables.py",
    "divs": _SCRIPTS_DIR / "content" / "format_div_spacing.py",
}


class FormatCommand:
    """Auto-format QMD content."""

    TARGETS = ["blanks", "python", "lists", "divs", "tables", "all"]

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def run(self, args: List[str]) -> bool:
        """Entry point — parse args and dispatch."""
        if not args or args[0] in ("-h", "--help"):
            self._print_help()
            return True

        target = args[0]
        if target not in self.TARGETS:
            console.print(f"[red]Unknown format target: {target}[/red]")
            self._print_help()
            return False

        # Remaining args are file paths or flags
        rest = args[1:]
        files, check_only = self._parse_rest(rest)

        if target == "all":
            return self._run_all(files, check_only)

        dispatch = {
            "blanks": self._run_blanks,
            "python": self._run_python,
            "lists": self._run_lists,
            "divs": self._run_divs,
            "tables": self._run_tables,
        }
        return dispatch[target](files, check_only)

    # ------------------------------------------------------------------
    # Argument helpers
    # ------------------------------------------------------------------

    def _parse_rest(self, rest: List[str]) -> tuple:
        """Parse remaining args into (file_list, check_only)."""
        check_only = False
        files: List[str] = []
        for arg in rest:
            if arg in ("--check", "-c"):
                check_only = True
            elif arg.startswith("-"):
                pass  # ignore unknown flags gracefully
            else:
                files.append(arg)
        return files, check_only

    def _resolve_files(self, file_args: List[str]) -> List[Path]:
        """Resolve file arguments to a list of QMD paths."""
        if file_args:
            result = []
            for f in file_args:
                p = Path(f)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if p.is_dir():
                    result.extend(sorted(p.rglob("*.qmd")))
                elif p.suffix == ".qmd" and p.exists():
                    result.append(p)
            return result
        # Default: all content files
        base = self.config_manager.book_dir / "contents"
        return sorted(base.rglob("*.qmd"))

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Target", style="cyan", width=12)
        table.add_column("Description", style="white", width=45)

        table.add_row("blanks", "Collapse extra blank lines (native)")
        table.add_row("python", "Format Python code blocks via Black (70 chars)")
        table.add_row("lists", "Fix bullet list spacing (blank line before lists)")
        table.add_row("divs", "Fix div/callout spacing (paragraph ↔ list gaps)")
        table.add_row("tables", "Prettify grid tables (align columns, bold headers)")
        table.add_row("all", "Run all formatters")

        console.print(Panel(table, title="binder format <target> [files...] [--check]", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder format blanks[/cyan]                [dim]# fix all files[/dim]")
        console.print("  [cyan]./binder format tables --check[/cyan]        [dim]# check only, no writes[/dim]")
        console.print("  [cyan]./binder format python path/to/ch.qmd[/cyan] [dim]# single file[/dim]")
        console.print()

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def _run_all(self, files: List[str], check_only: bool) -> bool:
        results = []
        for target in ("blanks", "lists", "divs", "python", "tables"):
            dispatch = {
                "blanks": self._run_blanks,
                "python": self._run_python,
                "lists": self._run_lists,
                "divs": self._run_divs,
                "tables": self._run_tables,
            }
            ok = dispatch[target](files, check_only)
            results.append((target, ok))

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Formatter", style="cyan")
        table.add_column("Status", style="white")
        for name, ok in results:
            status = "[green]PASS[/green]" if ok else "[red]MODIFIED[/red]"
            table.add_row(name, status)
        console.print(Panel(table, title="Binder Format Summary", border_style="cyan"))

        return all(ok for _, ok in results)

    # ------------------------------------------------------------------
    # Blanks  (native — ported from format_blank_lines.py)
    # ------------------------------------------------------------------

    def _run_blanks(self, file_args: List[str], check_only: bool) -> bool:
        """Collapse multiple consecutive blank lines into single blank lines."""
        qmd_files = self._resolve_files(file_args)
        modified = []

        for path in qmd_files:
            content = path.read_text(encoding="utf-8")
            new_content = self._collapse_blank_lines(content)
            if new_content != content:
                if not check_only:
                    path.write_text(new_content, encoding="utf-8")
                modified.append(path)

        if modified:
            label = "Would modify" if check_only else "Modified"
            console.print(f"[yellow]blanks: {label} {len(modified)} file(s)[/yellow]")
            for p in modified[:10]:
                console.print(f"  {self._rel(p)}")
            if len(modified) > 10:
                console.print(f"  [dim]... {len(modified) - 10} more[/dim]")
            return False  # pre-commit convention: modified = exit 1
        else:
            console.print("[green]blanks: All files clean[/green]")
            return True

    @staticmethod
    def _collapse_blank_lines(content: str) -> str:
        """Replace multiple consecutive blank lines with a single blank line.

        Preserves content inside code blocks.
        """
        lines = content.split("\n")
        result = []
        in_code_block = False
        blank_count = 0

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if blank_count > 0:
                    result.append("")
                    blank_count = 0
                result.append(line)
                continue

            if in_code_block:
                result.append(line)
                continue

            if line.strip() == "":
                blank_count += 1
            else:
                if blank_count > 0:
                    result.append("")
                    blank_count = 0
                result.append(line)

        if blank_count > 0:
            result.append("")

        return "\n".join(result)

    # ------------------------------------------------------------------
    # Lists  (native — ported from check_list_formatting.py)
    # ------------------------------------------------------------------

    def _run_lists(self, file_args: List[str], check_only: bool) -> bool:
        """Ensure blank line before bullet lists after colon-ending lines."""
        qmd_files = self._resolve_files(file_args)
        modified = []

        for path in qmd_files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except Exception:
                continue

            new_lines, changed = self._fix_list_spacing(lines)
            if changed:
                if not check_only:
                    path.write_text("".join(new_lines), encoding="utf-8")
                modified.append(path)

        if modified:
            label = "Would modify" if check_only else "Modified"
            console.print(f"[yellow]lists: {label} {len(modified)} file(s)[/yellow]")
            for p in modified[:10]:
                console.print(f"  {self._rel(p)}")
            if len(modified) > 10:
                console.print(f"  [dim]... {len(modified) - 10} more[/dim]")
            return False
        else:
            console.print("[green]lists: All files clean[/green]")
            return True

    @staticmethod
    def _fix_list_spacing(lines: List[str]) -> tuple:
        """Insert blank line between colon-ending paragraph and bullet list.

        Returns (new_lines, changed).
        """
        new_lines = []
        changed = False
        in_code_block = False

        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block

            new_lines.append(line)

            if in_code_block or i + 1 >= len(lines):
                continue

            next_line = lines[i + 1].rstrip()
            if (
                stripped
                and stripped.endswith(":")
                and not stripped.startswith("```")
                and not stripped.startswith(":::")
                and not stripped.startswith("#")
                and not stripped.startswith("|")
                and next_line.startswith("- ")
            ):
                new_lines.append("\n")
                changed = True

        return new_lines, changed

    # ------------------------------------------------------------------
    # Python  (delegates to format_python_in_qmd.py)
    # ------------------------------------------------------------------

    def _run_python(self, file_args: List[str], check_only: bool) -> bool:
        """Format Python code blocks using Black."""
        script = _SCRIPT_PATHS["python"]
        if not script.exists():
            console.print(f"[red]python: Script not found: {script}[/red]")
            return False

        qmd_files = self._resolve_files(file_args)
        if not qmd_files:
            console.print("[green]python: No files to process[/green]")
            return True

        cmd = [sys.executable, str(script)] + [str(f) for f in qmd_files]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]python: All files clean[/green]")
            return True
        else:
            if result.stdout:
                for line in result.stdout.strip().splitlines()[:10]:
                    console.print(f"  [yellow]{line}[/yellow]")
            console.print(f"[yellow]python: {len(qmd_files)} file(s) processed[/yellow]")
            return False

    # ------------------------------------------------------------------
    # Divs  (delegates to format_div_spacing.py)
    # ------------------------------------------------------------------

    def _run_divs(self, file_args: List[str], check_only: bool) -> bool:
        """Fix div/callout spacing."""
        script = _SCRIPT_PATHS["divs"]
        if not script.exists():
            console.print(f"[red]divs: Script not found: {script}[/red]")
            return False

        qmd_files = self._resolve_files(file_args)
        if not qmd_files:
            console.print("[green]divs: No files to process[/green]")
            return True

        modified_count = 0
        for path in qmd_files:
            mode = "--check" if check_only else "-f"
            cmd = [sys.executable, str(script), mode, str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                modified_count += 1

        if modified_count:
            label = "Would modify" if check_only else "Modified"
            console.print(f"[yellow]divs: {label} {modified_count} file(s)[/yellow]")
            return False
        else:
            console.print("[green]divs: All files clean[/green]")
            return True

    # ------------------------------------------------------------------
    # Tables  (delegates to format_tables.py)
    # ------------------------------------------------------------------

    def _run_tables(self, file_args: List[str], check_only: bool) -> bool:
        """Prettify grid tables."""
        script = _SCRIPT_PATHS["tables"]
        if not script.exists():
            console.print(f"[red]tables: Script not found: {script}[/red]")
            return False

        mode = "--check" if check_only else "--fix"
        if file_args:
            cmd = [sys.executable, str(script), mode]
            for f in file_args:
                cmd.extend(["-f", f])
        else:
            content_dir = self.config_manager.book_dir / "contents"
            cmd = [sys.executable, str(script), mode, "-d", str(content_dir)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]tables: All files clean[/green]")
            return True
        else:
            if result.stdout:
                for line in result.stdout.strip().splitlines()[:10]:
                    console.print(f"  [yellow]{line}[/yellow]")
            console.print(f"[yellow]tables: Issues found (exit {result.returncode})[/yellow]")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        """Return path relative to book dir for display."""
        try:
            return str(path.relative_to(self.config_manager.book_dir))
        except ValueError:
            return str(path)
