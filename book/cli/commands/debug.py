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
LEGACY_DEBUG_LOG_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tools" / "scripts" / "testing" / "logs"
)


def _assimilate_legacy_debug_logs(book_dir: Path) -> List[Tuple[Path, Path]]:
    """Move legacy debug logs into the new _build/debug structure.

    Returns:
        List of (source_path, destination_path) moves performed.
    """
    if not LEGACY_DEBUG_LOG_ROOT.exists():
        return []

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    migrated: List[Tuple[Path, Path]] = []
    legacy_target_root = book_dir / "_build" / "debug" / "_legacy"

    for volume_dir in sorted(LEGACY_DEBUG_LOG_ROOT.glob("vol*")):
        legacy_debug_dir = volume_dir / "debug"
        if not legacy_debug_dir.exists():
            continue
        if not any(legacy_debug_dir.rglob("*.log")):
            continue

        destination = legacy_target_root / volume_dir.name / f"migrated-{timestamp}"
        suffix = 1
        while destination.exists():
            destination = legacy_target_root / volume_dir.name / f"migrated-{timestamp}-{suffix}"
            suffix += 1

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_debug_dir), str(destination))
        migrated.append((legacy_debug_dir, destination))

    return migrated



def _find_chapter_qmd(book_dir: Path, chapter: str, volume: str) -> Path:
    """Locate the .qmd file for a chapter.

    Searches the volume directory first, then falls back to the shared
    directory (e.g. contents/shared/notation.qmd).
    """
    contents_dir = book_dir / "contents"
    # Search volume dir and shared dir (covers frontmatter, parts, shared files)
    for matches in [
        list((contents_dir / volume).rglob(f"{chapter}.qmd")),
        list((contents_dir / "shared").rglob(f"{chapter}.qmd")),
    ]:
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Chapter '{chapter}' not found under {contents_dir / volume} "
        f"or {contents_dir / 'shared'}"
    )


def _get_output_dir(book_dir: Path, format_type: str, volume: str) -> Optional[Path]:
    """Return the build output directory (same for all formats: PDF, EPUB, HTML)."""
    if format_type in ("pdf", "epub", "html"):
        return book_dir / "_build" / f"{format_type}-{volume}"
    return None


def _safe_artifact_stem(stem: str) -> str:
    """Convert arbitrary labels into filesystem-safe artifact names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "artifact"


def _print_step_result(success: bool, duration: float, warnings: List[str]) -> None:
    """Print PASS/WARN status and any Quarto warnings for a build step."""
    if warnings:
        console.print(f"[yellow]WARN[/yellow] ({duration:.1f}s)")
        for w in warnings:
            console.print(f"         [yellow]⚠ {w}[/yellow]")
    else:
        console.print(f"[green]PASS[/green] ({duration:.1f}s)")


# Quarto warning patterns to surface even when the build succeeds.
# Each entry is (pattern, human_label).
_QUARTO_WARN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"Duplicate note reference '([^']+)'", re.IGNORECASE),
        "Duplicate footnote reference",
    ),
    (
        re.compile(r"The following string was found in the document: :::", re.IGNORECASE),
        "Unclosed/stray fenced div (:::)",
    ),
]


def _extract_quarto_warnings(output: str) -> List[str]:
    """Scan build output for known Quarto warning patterns.

    Returns a deduplicated list of human-readable warning strings.
    """
    found: List[str] = []
    seen: set = set()
    for pattern, label in _QUARTO_WARN_PATTERNS:
        for m in pattern.finditer(output):
            # Include the captured group (e.g., fn ID) if present
            detail = m.group(1) if m.lastindex else ""
            msg = f"{label}: {detail}" if detail else label
            if msg not in seen:
                seen.add(msg)
                found.append(msg)
    return found


def _build_and_check(
    book_dir: Path,
    chapter_name: str,
    volume: str,
    format_type: str,
    log_file: Path,
    verbose: bool = False,
    artifact_dir: Optional[Path] = None,
    artifact_stem: Optional[str] = None,
) -> Tuple[bool, float, str, List[str]]:
    """Run a single chapter build and check if output was created.

    Returns:
        (success, duration_seconds, error_snippet, quarto_warnings)
    """
    cmd = [
        "./binder",
        "build",
        format_type,
        chapter_name,
        f"--{volume}",
        "-v",
    ]

    output_dir = _get_output_dir(book_dir, format_type, volume)
    if not output_dir:
        log_file.write_text("Unknown format type\n")
        return False, 0.0, "Unknown format type", []

    # Delete previous output to ensure clean test (same rule: any .pdf, any .epub, index.html)
    from cli.core.config import get_output_file
    existing = get_output_file(output_dir, format_type)
    if existing is not None:
        try:
            existing.unlink()
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

        # Success = output file present (any .pdf, any .epub, or index.html)
        resolved = get_output_file(output_dir, format_type)
        combined = result.stdout + result.stderr
        quarto_warnings = _extract_quarto_warnings(combined)

        if resolved is not None and resolved.exists():
            if artifact_dir is not None:
                try:
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    stem = _safe_artifact_stem(artifact_stem or chapter_name)
                    artifact_ext = resolved.suffix or ".artifact"
                    artifact_path = artifact_dir / f"{stem}{artifact_ext}"
                    shutil.copy2(resolved, artifact_path)
                    console.print(f"[dim]Saved debug artifact: {artifact_path}[/dim]")
                except Exception as exc:
                    console.print(
                        f"[yellow]⚠️ Failed to save debug artifact for {chapter_name}: {exc}[/yellow]"
                    )
            return True, duration, "", quarto_warnings
        else:
            error_lines = combined.strip().split("\n")[-20:]
            return False, duration, "\n".join(error_lines), quarto_warnings

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        log_file.write_text(f"TIMEOUT after {duration:.0f}s\nCommand: {' '.join(cmd)}")
        return False, duration, "TIMEOUT: Build exceeded 10 minutes", []
    except Exception as e:
        duration = time.time() - start
        log_file.write_text(f"EXCEPTION: {e}\nCommand: {' '.join(cmd)}")
        return False, duration, f"EXCEPTION: {e}", []


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

        migrated_legacy = _assimilate_legacy_debug_logs(self.book_dir)
        if migrated_legacy:
            console.print(
                f"[dim]Migrated {len(migrated_legacy)} legacy debug log folder(s) "
                "to book/quarto/_build/debug/_legacy/[/dim]"
            )

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = (
            self.book_dir
            / "_build" / "debug"
            / volume / format_type
            / run_id
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Debug logs: {log_dir}[/dim]")

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
        chapters = self.chapter_discovery.get_chapters_from_config(volume)

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

            success, duration, error, warnings = _build_and_check(
                self.book_dir,
                chapter_name,
                volume,
                format_type,
                log_file,
                self.verbose,
                artifact_dir=log_dir / "phase1" / "artifacts",
                artifact_stem=f"{i:02d}_{chapter_name}",
            )

            if success:
                if warnings:
                    console.print(f" [yellow]WARN[/yellow] ({duration:.1f}s)")
                    for w in warnings:
                        console.print(f"         [yellow]⚠ {w}[/yellow]")
                else:
                    console.print(f" [green]PASS[/green] ({duration:.1f}s)")
                passed += 1
            else:
                console.print(f" [red]FAIL[/red] ({duration:.1f}s)")
                failures.append((chapter_name, error))
                if warnings:
                    for w in warnings:
                        console.print(f"         [yellow]⚠ {w}[/yellow]")
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
        success, duration, _, warnings = _build_and_check(
            self.book_dir,
            chapter_name,
            volume,
            format_type,
            log_file,
            self.verbose,
            artifact_dir=log_dir / "artifacts",
            artifact_stem="preamble",
        )
        if not success:
            console.print(f"[red]FAIL[/red] ({duration:.1f}s)")
            return -1
        _print_step_result(success, duration, warnings)

        # Step 2: Test full chapter (confirm it actually fails)
        console.print(f"  [dim][full][/dim] All {num_sections} sections", end="  ")
        content = self._assemble_content(chapter, num_sections - 1)
        qmd_path.write_text(content, encoding="utf-8")
        log_file = log_dir / "binary_full.log"
        success, duration, _, warnings = _build_and_check(
            self.book_dir,
            chapter_name,
            volume,
            format_type,
            log_file,
            self.verbose,
            artifact_dir=log_dir / "artifacts",
            artifact_stem="full",
        )
        if success:
            _print_step_result(success, duration, warnings)
            return None
        console.print(f"[red]FAIL[/red] ({duration:.1f}s)")
        for w in warnings:
            console.print(f"         [yellow]⚠ {w}[/yellow]")

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
            success, duration, _, warnings = _build_and_check(
                self.book_dir,
                chapter_name,
                volume,
                format_type,
                log_file,
                self.verbose,
                artifact_dir=log_dir / "artifacts",
                artifact_stem=f"step_{build_count:02d}_upto_{mid}",
            )

            if success:
                _print_step_result(success, duration, warnings)
                lo = mid + 1
            else:
                console.print(f"[red]FAIL[/red] ({duration:.1f}s)")
                for w in warnings:
                    console.print(f"         [yellow]⚠ {w}[/yellow]")
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
