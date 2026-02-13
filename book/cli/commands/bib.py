"""
``binder bib`` — Bibliography management.

Subcommands:
    list    — Show all .bib files with entry counts
    clean   — Remove unused entries from .bib files
    update  — Run betterbib update -i on .bib files (fetch proper metadata)
    sync    — Clean then update (full pipeline)
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class BibCommand:
    """Native ``binder bib`` command group."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, args: List[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder bib",
            description="Bibliography management",
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=["list", "clean", "update", "sync"],
            help="Subcommand to run",
        )
        parser.add_argument("--path", default=None, help="File or directory")
        parser.add_argument("--vol1", action="store_true", help="Volume I only")
        parser.add_argument("--vol2", action="store_true", help="Volume II only")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would change without modifying files",
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

        if ns.subcommand == "list":
            return self._run_list(root)
        elif ns.subcommand == "clean":
            return self._run_clean(root, ns.dry_run)
        elif ns.subcommand == "update":
            return self._run_update(root)
        elif ns.subcommand == "sync":
            return self._run_sync(root, ns.dry_run)
        return False

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Subcommand", style="cyan", width=14)
        table.add_column("Description", style="white", width=50)
        table.add_row("list", "Show all .bib files with entry counts")
        table.add_row("clean", "Remove unused entries from .bib files")
        table.add_row("update", "Run betterbib update -i (fetch proper metadata)")
        table.add_row("sync", "Clean then update (full pipeline)")
        console.print(Panel(table, title="binder bib <subcommand>", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder bib list --vol1[/cyan]")
        console.print("  [cyan]./binder bib clean --vol1 --dry-run[/cyan]")
        console.print("  [cyan]./binder bib update --vol1[/cyan]")
        console.print("  [cyan]./binder bib sync --vol1[/cyan]")
        console.print()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, path_arg: Optional[str], vol1: bool, vol2: bool) -> Path:
        if path_arg:
            p = Path(path_arg)
            return p if p.is_absolute() else Path.cwd() / p
        base = self.config_manager.book_dir / "contents"
        if vol1:
            return base / "vol1"
        if vol2:
            return base / "vol2"
        return base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_bib_files(self, root: Path) -> List[Path]:
        """Find all .bib files under root."""
        if root.is_file() and root.suffix == ".bib":
            return [root]
        return sorted(root.rglob("*.bib"))

    def _find_qmd_files(self, root: Path) -> List[Path]:
        """Find all .qmd files under root."""
        if root.is_file():
            return [root] if root.suffix == ".qmd" else []
        return sorted(root.rglob("*.qmd"))

    def _count_entries(self, bib_path: Path) -> int:
        """Count @type{key, entries in a .bib file."""
        try:
            content = bib_path.read_text(encoding="utf-8")
            return len(re.findall(r"^@\w+\{", content, re.MULTILINE))
        except Exception:
            return 0

    def _extract_cited_keys(self, qmd_path: Path) -> Set[str]:
        """Extract all citation keys from a .qmd file."""
        try:
            content = qmd_path.read_text(encoding="utf-8")
        except Exception:
            return set()
        # Standard @key citations
        keys = set(re.findall(r"@([\w\d:\-_.]+)", content))
        # Filter out cross-ref prefixes
        keys = {
            k.rstrip(".,;:")
            for k in keys
            if not k.startswith(("fig-", "tbl-", "sec-", "eq-", "lst-"))
        }
        return keys

    def _extract_bib_keys(self, bib_path: Path) -> Set[str]:
        """Extract all entry keys from a .bib file."""
        try:
            content = bib_path.read_text(encoding="utf-8")
            return set(re.findall(r"^@\w+\{([^,\s]+)", content, re.MULTILINE))
        except Exception:
            return set()

    def _bib_for_qmd(self, qmd_path: Path) -> Optional[Path]:
        """Find the .bib file referenced in a .qmd file's YAML front matter."""
        try:
            content = qmd_path.read_text(encoding="utf-8")
        except Exception:
            return None
        m = re.search(r"^bibliography:\s*([\w\-.]+\.bib)\s*$", content, re.MULTILINE)
        if m:
            bib = qmd_path.parent / m.group(1)
            return bib if bib.exists() else None
        return None

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config_manager.book_dir))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def _run_list(self, root: Path) -> bool:
        """Show all .bib files with entry counts."""
        bib_files = self._find_bib_files(root)
        if not bib_files:
            console.print("[yellow]No .bib files found.[/yellow]")
            return True

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("File", style="cyan", width=50)
        table.add_column("Entries", style="white", width=8, justify="right")

        total = 0
        for bib in bib_files:
            count = self._count_entries(bib)
            total += count
            table.add_row(self._relative(bib), str(count))

        console.print(Panel(table, title="Bibliography Files", border_style="cyan"))
        console.print(
            f"[bold]Total:[/bold] {len(bib_files)} files, {total} entries"
        )
        return True

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------

    def _run_clean(self, root: Path, dry_run: bool = False) -> bool:
        """Remove unused entries from .bib files."""
        qmd_files = self._find_qmd_files(root)
        if not qmd_files:
            console.print("[yellow]No QMD files found.[/yellow]")
            return False

        # Collect ALL cited keys from all QMD files in scope
        all_cited: Set[str] = set()
        for qmd in qmd_files:
            all_cited.update(self._extract_cited_keys(qmd))

        # Find all .bib files in scope — bibliography is declared at project
        # level in Quarto YAML configs, not in individual QMD front matter.
        # So we match all bib files under root against all citations under root.
        bib_files = self._find_bib_files(root)

        # Also check per-QMD bibliography declarations (some chapters have their own)
        for qmd in qmd_files:
            bib = self._bib_for_qmd(qmd)
            if bib and bib not in bib_files:
                bib_files.append(bib)

        if not bib_files:
            console.print("[yellow]No .bib files found.[/yellow]")
            return True

        # Build mapping: each bib file checked against all cited keys
        bib_cited: Dict[Path, Set[str]] = {bib: all_cited for bib in bib_files}

        # Warn if scoped to a narrow path — citations from other chapters
        # won't be visible, which could cause false positives.
        contents_dir = self.config_manager.book_dir / "contents"
        is_narrow = root != contents_dir and not any(
            root == contents_dir / v for v in ("vol1", "vol2")
        )
        if is_narrow:
            console.print(
                "[yellow]⚠ Narrow scope:[/yellow] Only citations from QMD files "
                f"under [cyan]{self._relative(root)}[/cyan] are visible.\n"
                "  Entries cited by other chapters may appear unused. "
                "Use [cyan]--vol1[/cyan] or [cyan]--vol2[/cyan] for safe cleaning.\n"
            )

        console.print(
            f"[dim]Collected {len(all_cited)} citation keys from "
            f"{len(qmd_files)} QMD files[/dim]\n"
        )

        total_removed = 0
        for bib_path, cited_keys in sorted(bib_cited.items()):
            bib_keys = self._extract_bib_keys(bib_path)
            unused = bib_keys - cited_keys
            rel = self._relative(bib_path)

            if not unused:
                console.print(f"  [green]✓[/green] {rel}: all {len(bib_keys)} entries used")
                continue

            total_removed += len(unused)
            if dry_run:
                console.print(
                    f"  [yellow]⚠[/yellow] {rel}: {len(unused)} unused of {len(bib_keys)} "
                    f"[dim](dry run — no changes)[/dim]"
                )
                for key in sorted(unused)[:10]:
                    console.print(f"      [dim]- {key}[/dim]")
                if len(unused) > 10:
                    console.print(f"      [dim]... and {len(unused) - 10} more[/dim]")
            else:
                # Remove unused entries by rewriting the file
                self._remove_entries(bib_path, unused)
                console.print(
                    f"  [green]✓[/green] {rel}: removed {len(unused)} unused entries "
                    f"({len(bib_keys) - len(unused)} remaining)"
                )

        action = "would remove" if dry_run else "removed"
        console.print(f"\n[bold]Total:[/bold] {action} {total_removed} unused entries from {len(bib_cited)} files")
        return True

    def _remove_entries(self, bib_path: Path, keys_to_remove: Set[str]) -> None:
        """Remove specific entries from a .bib file, preserving formatting."""
        content = bib_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        output: List[str] = []
        skip = False
        brace_depth = 0

        for line in lines:
            # Detect entry start
            m = re.match(r"^@\w+\{([^,\s]+)", line)
            if m:
                key = m.group(1)
                if key in keys_to_remove:
                    skip = True
                    brace_depth = line.count("{") - line.count("}")
                    continue
                else:
                    skip = False

            if skip:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    skip = False
                continue

            output.append(line)

        # Clean up multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(output))
        bib_path.write_text(cleaned.strip() + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _run_update(self, root: Path) -> bool:
        """Run betterbib update -i on .bib files."""
        # Check if betterbib is available
        try:
            subprocess.run(
                ["betterbib", "--version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            console.print(
                "[red]betterbib not found.[/red] Install with: "
                "[cyan]pip install betterbib[/cyan]"
            )
            return False

        bib_files = self._find_bib_files(root)
        if not bib_files:
            console.print("[yellow]No .bib files found.[/yellow]")
            return True

        console.print(f"[bold]Updating {len(bib_files)} .bib files with betterbib...[/bold]\n")

        success = 0
        failed = 0
        for bib in bib_files:
            rel = self._relative(bib)
            console.print(f"  Updating {rel}...", end=" ")
            try:
                result = subprocess.run(
                    ["betterbib", "update", "-i", str(bib)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    console.print("[green]✓[/green]")
                    success += 1
                else:
                    console.print(f"[red]✗[/red] [dim]{result.stderr.strip()[:80]}[/dim]")
                    failed += 1
            except subprocess.TimeoutExpired:
                console.print("[red]✗ timed out[/red]")
                failed += 1

        console.print(f"\n[bold]Done:[/bold] {success} updated, {failed} failed")
        return failed == 0

    # ------------------------------------------------------------------
    # Sync (clean + update)
    # ------------------------------------------------------------------

    def _run_sync(self, root: Path, dry_run: bool = False) -> bool:
        """Full pipeline: clean unused entries, then update with betterbib."""
        console.print("[bold cyan]Step 1/2: Cleaning unused entries...[/bold cyan]\n")
        clean_ok = self._run_clean(root, dry_run)

        if dry_run:
            console.print(
                "\n[yellow]Dry run — skipping betterbib update. "
                "Run without --dry-run to apply changes and update.[/yellow]"
            )
            return clean_ok

        console.print("\n[bold cyan]Step 2/2: Updating with betterbib...[/bold cyan]\n")
        update_ok = self._run_update(root)

        return clean_ok and update_ok
