"""
Debug command implementation for MLSysBook CLI.

Two-phase build debugger that isolates build failures:
  Phase 1: Scan chapters one-by-one to find which chapter(s) fail
  Phase 2: Binary search within a failing chapter to find the exact section

Usage via binder:
    ./binder debug pdf --vol1                    # Full debug: find chapter + section
    ./binder debug pdf --vol1 --chapter training # Skip to section-level debug
    ./binder debug html --vol2 -v                # Verbose output
"""

import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Path to section_splitter (in content scripts)
CONTENT_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "tools" / "scripts" / "content"


def _get_chapters_from_config(book_dir: Path, volume: str) -> List[str]:
    """Read ordered chapter list from the PDF config file.

    Extracts chapter names from the volume's PDF config, excluding
    frontmatter, backmatter, parts, and other non-chapter files.
    """
    config_file = book_dir / "config" / f"_quarto-pdf-{volume}.yml"

    if not config_file.exists():
        return []

    content = config_file.read_text()

    # Match chapter paths like: - contents/vol1/training/training.qmd
    # Also matches commented-out lines (# - contents/...)
    pattern = rf'#?\s*-\s*contents/{volume}/[^/]+/([^/]+)\.qmd'

    exclude = {
        "index", "references", "glossary", "foreword", "about",
        "acknowledgements", "foundations_principles", "build_principles",
        "optimize_principles", "deploy_principles", "inference_principles",
        "infrastructure_principles", "production_principles",
        "responsible_principles", "scale_principles",
    }

    chapters = []
    for match in re.finditer(pattern, content):
        chapter = match.group(1)
        if chapter not in chapters and chapter not in exclude:
            chapters.append(chapter)

    return chapters


def _get_chapters_from_directory(book_dir: Path, volume: str) -> List[str]:
    """Fallback: scan filesystem for chapter directories."""
    contents_dir = book_dir / "contents" / volume

    if not contents_dir.exists():
        return []

    exclude = {"parts", "frontmatter", "backmatter", "index", "glossary"}

    chapters = []
    for item in sorted(contents_dir.iterdir()):
        if item.is_dir() and item.name not in exclude:
            qmd_file = item / f"{item.name}.qmd"
            if qmd_file.exists():
                chapters.append(item.name)

    return chapters


def _find_chapter_qmd(book_dir: Path, chapter: str, volume: str) -> Path:
    """Locate the .qmd file for a chapter."""
    contents_dir = book_dir / "contents" / volume
    for subdir in contents_dir.iterdir():
        if subdir.is_dir():
            qmd = subdir / f"{chapter}.qmd"
            if qmd.exists():
                return qmd
    raise FileNotFoundError(f"Chapter '{chapter}' not found in {contents_dir}")


def _get_output_path(book_dir: Path, format_type: str, volume: str) -> Optional[Path]:
    """Get the expected output file path for a build."""
    if format_type == "pdf":
        return book_dir / "_build" / f"pdf-{volume}" / "Machine-Learning-Systems.pdf"
    elif format_type == "epub":
        return book_dir / "_build" / f"epub-{volume}" / "Machine-Learning-Systems.epub"
    elif format_type == "html":
        return book_dir / "_build" / f"html-{volume}" / "index.html"
    return None


def _build_and_check(
    book_dir: Path,
    chapter_name: str,
    volume: str,
    format_type: str,
    log_file: Path,
    verbose: bool = False,
) -> Tuple[bool, float, str]:
    """Run a single chapter build and check if output was created.

    Returns:
        (success, duration_seconds, error_snippet)
    """
    cmd = [
        "./binder",
        format_type,
        chapter_name,
        f"--{volume}",
        "-v",
    ]

    output_path = _get_output_path(book_dir, format_type, volume)

    # Delete previous output to ensure clean test
    if output_path and output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=book_dir.parent,  # Run from book/ directory
            capture_output=True,
            text=True,
            timeout=600,
        )
        duration = time.time() - start

        # Write full log
        full_output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        full_output += f"=== TIMESTAMP ===\n{datetime.now().isoformat()}\n\n"
        full_output += f"=== STDOUT ===\n{result.stdout}\n\n"
        full_output += f"=== STDERR ===\n{result.stderr}\n\n"
        full_output += f"=== EXIT CODE ===\n{result.returncode}\n"
        full_output += f"=== DURATION ===\n{duration:.1f}s\n"
        log_file.write_text(full_output)

        if output_path and output_path.exists():
            return True, duration, ""
        else:
            combined = result.stdout + result.stderr
            error_lines = combined.strip().split("\n")[-20:]
            return False, duration, "\n".join(error_lines)

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        log_file.write_text(f"TIMEOUT after {duration:.0f}s\nCommand: {' '.join(cmd)}")
        return False, duration, "TIMEOUT: Build exceeded 10 minutes"
    except Exception as e:
        duration = time.time() - start
        log_file.write_text(f"EXCEPTION: {e}\nCommand: {' '.join(cmd)}")
        return False, duration, f"EXCEPTION: {e}"


class DebugCommand:
    """Two-phase build debugger for MLSysBook."""

    def __init__(self, config_manager, chapter_discovery, verbose: bool = False):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.verbose = verbose
        self.book_dir = config_manager.book_dir

    def debug_build(
        self,
        format_type: str,
        volume: str,
        chapter: Optional[str] = None,
    ) -> bool:
        """Main entry point for the debug command.

        Args:
            format_type: Build format (pdf, html, epub)
            volume: Volume to debug (vol1, vol2)
            chapter: If provided, skip to section-level debug for this chapter

        Returns:
            True if debugging completed (regardless of findings)
        """
        volume_label = "Volume I" if volume == "vol1" else "Volume II"

        banner = Panel(
            f"[bold red]Build Debugger[/bold red]\n"
            f"[dim]{volume_label} / {format_type.upper()}[/dim]",
            border_style="red",
        )
        console.print(banner)

        log_dir = (
            Path(__file__).resolve().parents[2]
            / "tools" / "scripts" / "testing" / "logs"
            / volume / "debug"
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        if chapter:
            # Skip Phase 1, go directly to section debug
            console.print(f"\n[bold]Skipping to Phase 2: section-level debug for [cyan]{chapter}[/cyan][/bold]\n")
            return self._phase2_section_debug(chapter, volume, format_type, log_dir)
        else:
            # Phase 1: find failing chapters
            console.print("\n[bold]Phase 1:[/bold] Scanning chapters for build failures...\n")
            failures = self._phase1_chapter_scan(volume, format_type, log_dir)

            if not failures:
                console.print(Panel(
                    "[bold green]All chapters build successfully.[/bold green]\n"
                    "[dim]No failures to debug.[/dim]",
                    border_style="green",
                ))
                return True

            # Report Phase 1 results
            console.print(f"\n[bold red]Found {len(failures)} failing chapter(s):[/bold red]")
            for ch_name, error in failures:
                console.print(f"  [red]x[/red] {ch_name}")

            # Phase 2: drill into each failure
            for i, (ch_name, _) in enumerate(failures):
                console.print(f"\n[bold]Phase 2 ({i+1}/{len(failures)}):[/bold] "
                              f"Section-level debug for [cyan]{ch_name}[/cyan]\n")
                self._phase2_section_debug(ch_name, volume, format_type, log_dir)

            return True

    def _phase1_chapter_scan(
        self,
        volume: str,
        format_type: str,
        log_dir: Path,
    ) -> List[Tuple[str, str]]:
        """Phase 1: Build each chapter individually, collect failures.

        Returns:
            List of (chapter_name, error_snippet) for each failure
        """
        chapters = _get_chapters_from_config(self.book_dir, volume)
        if not chapters:
            console.print("[yellow]Config not found, falling back to directory scan...[/yellow]")
            chapters = _get_chapters_from_directory(self.book_dir, volume)

        if not chapters:
            console.print("[red]No chapters found.[/red]")
            return []

        console.print(f"[dim]Testing {len(chapters)} chapters...[/dim]\n")

        failures = []
        passed = 0

        for i, chapter_name in enumerate(chapters, 1):
            console.print(
                f"  [{i:2d}/{len(chapters)}] {chapter_name:<30s}",
                end="",
            )

            chapter_log_dir = log_dir / "phase1"
            chapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = chapter_log_dir / f"{chapter_name}.log"

            success, duration, error = _build_and_check(
                self.book_dir, chapter_name, volume, format_type, log_file, self.verbose
            )

            if success:
                console.print(f" [green]PASS[/green] ({duration:.1f}s)")
                passed += 1
            else:
                console.print(f" [red]FAIL[/red] ({duration:.1f}s)")
                failures.append((chapter_name, error))
                if self.verbose and error:
                    for line in error.strip().split("\n")[-3:]:
                        console.print(f"         [dim]{line}[/dim]")

        console.print(f"\n[dim]Results: {passed} passed, {len(failures)} failed[/dim]")
        return failures

    def _phase2_section_debug(
        self,
        chapter_name: str,
        volume: str,
        format_type: str,
        log_dir: Path,
    ) -> bool:
        """Phase 2: Binary search within a chapter to find the failing section.

        Returns:
            True if debugging completed
        """
        # Import section_splitter
        sys.path.insert(0, str(CONTENT_SCRIPTS_DIR))
        try:
            from section_splitter import split_chapter
        except ImportError:
            console.print("[red]Cannot import section_splitter.py from content scripts.[/red]")
            console.print(f"[dim]Expected at: {CONTENT_SCRIPTS_DIR / 'section_splitter.py'}[/dim]")
            return False

        # Find the chapter .qmd file
        try:
            qmd_path = _find_chapter_qmd(self.book_dir, chapter_name, volume)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return False

        # Parse into sections
        console.print(f"[dim]Parsing {qmd_path.name} into sections...[/dim]")
        chapter = split_chapter(str(qmd_path))
        num_sections = len(chapter.sections)

        if num_sections == 0:
            console.print("[yellow]No ## sections found in chapter.[/yellow]")
            return False

        # Show section map
        section_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        section_table.add_column("#", style="dim", width=4)
        section_table.add_column("Section", width=50)
        section_table.add_column("Lines", style="dim", width=12)
        for sec in chapter.sections:
            section_table.add_row(
                str(sec.index),
                sec.title,
                f"L{sec.start_line}-{sec.end_line}",
            )
        console.print(section_table)
        console.print()

        # Set up log directory for section debug
        section_log_dir = log_dir / "phase2" / chapter_name
        section_log_dir.mkdir(parents=True, exist_ok=True)

        # Back up original file
        backup_path = qmd_path.with_suffix(".qmd.debug_backup")
        shutil.copy2(qmd_path, backup_path)

        def restore_original(signum=None, frame=None):
            if backup_path.exists():
                shutil.copy2(backup_path, qmd_path)
                backup_path.unlink()
            if signum is not None:
                console.print("\n[yellow]Interrupted. Original file restored.[/yellow]")
                sys.exit(1)

        # Register signal handlers for safe cleanup
        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, restore_original)
        signal.signal(signal.SIGTERM, restore_original)

        try:
            failing_idx = self._binary_search_sections(
                chapter_name, volume, format_type, chapter,
                qmd_path, section_log_dir,
            )
        finally:
            restore_original()
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

        # Report results
        console.print()
        if failing_idx is None:
            console.print(Panel(
                f"[bold green]All {num_sections} sections in {chapter_name} build successfully.[/bold green]",
                border_style="green",
            ))
        elif failing_idx == -1:
            console.print(Panel(
                "[bold red]Preamble itself fails to build.[/bold red]\n"
                "[dim]The issue is in the YAML frontmatter or content before the first ## section.[/dim]",
                border_style="red",
            ))
        else:
            sec = chapter.sections[failing_idx]
            result_text = Text()
            result_text.append("Build breaks at section ", style="bold red")
            result_text.append(f"{failing_idx}", style="bold cyan")
            result_text.append(": ", style="bold red")
            result_text.append(f'"{sec.title}"', style="bold white")
            result_text.append(f"\n\nFile:   {qmd_path.name}", style="dim")
            result_text.append(f"\nLines:  {sec.start_line}-{sec.end_line}", style="dim")
            if sec.section_id:
                result_text.append(f"\nID:     #{sec.section_id}", style="dim")
            result_text.append(f"\nLogs:   {section_log_dir}", style="dim")

            console.print(Panel(result_text, title="Result", border_style="red"))

        return True

    def _binary_search_sections(
        self,
        chapter_name: str,
        volume: str,
        format_type: str,
        chapter,  # ChapterStructure
        qmd_path: Path,
        log_dir: Path,
    ) -> Optional[int]:
        """Binary search for the first section that breaks the build.

        Returns:
            Section index that causes failure, -1 if preamble fails, None if all pass.
        """
        num_sections = len(chapter.sections)

        # Step 1: Test preamble only
        console.print(f"  [dim][pre][/dim]  Preamble only", end="  ")
        content = self._assemble_content(chapter, -1)
        qmd_path.write_text(content, encoding="utf-8")
        log_file = log_dir / "binary_preamble.log"
        success, duration, _ = _build_and_check(
            self.book_dir, chapter_name, volume, format_type, log_file, self.verbose
        )
        if not success:
            console.print(f"[red]FAIL[/red] ({duration:.1f}s)")
            return -1
        console.print(f"[green]PASS[/green] ({duration:.1f}s)")

        # Step 2: Test full chapter (confirm it actually fails)
        console.print(f"  [dim][full][/dim] All {num_sections} sections", end="  ")
        content = self._assemble_content(chapter, num_sections - 1)
        qmd_path.write_text(content, encoding="utf-8")
        log_file = log_dir / "binary_full.log"
        success, duration, _ = _build_and_check(
            self.book_dir, chapter_name, volume, format_type, log_file, self.verbose
        )
        if success:
            console.print(f"[green]PASS[/green] ({duration:.1f}s)")
            return None
        console.print(f"[red]FAIL[/red] ({duration:.1f}s)")

        # Step 3: Binary search
        lo, hi = 0, num_sections - 1
        build_count = 2

        while lo < hi:
            mid = (lo + hi) // 2
            sec = chapter.sections[mid]
            build_count += 1

            label = f'Up to #{mid}: "{sec.title[:40]}"'
            console.print(f"  [dim][bisect {build_count}][/dim] {label}", end="  ")

            content = self._assemble_content(chapter, mid)
            qmd_path.write_text(content, encoding="utf-8")
            log_file = log_dir / f"binary_step_{build_count:02d}_upto_{mid}.log"
            success, duration, _ = _build_and_check(
                self.book_dir, chapter_name, volume, format_type, log_file, self.verbose
            )

            if success:
                console.print(f"[green]PASS[/green] ({duration:.1f}s)")
                lo = mid + 1
            else:
                console.print(f"[red]FAIL[/red] ({duration:.1f}s)")
                hi = mid

        console.print(f"\n[dim]Isolated in {build_count} builds (binary search).[/dim]")
        return lo

    @staticmethod
    def _assemble_content(chapter, up_to_section: int) -> str:
        """Build chapter content including sections 0..up_to_section.

        Args:
            chapter: Parsed ChapterStructure
            up_to_section: Include sections with index <= this value (-1 = preamble only)
        """
        parts = []

        if chapter.frontmatter:
            parts.append(chapter.frontmatter)

        if chapter.pre_content:
            parts.append(chapter.pre_content)

        for section in chapter.sections:
            if section.index <= up_to_section:
                parts.append(section.content)

        return "\n".join(parts)
