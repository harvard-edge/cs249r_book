"""Isolate build failures chapter by chapter, then section by section.

``binder debug <fmt> --volN`` runs two phases:

1. **Chapter scan.** Every chapter in the volume's PDF order is built on its
   own. With ``--parallel N`` the builds run N at a time.
2. **Section bisection.** Each failing chapter (or the one named with
   ``--chapter``) is truncated to a growing number of ``##`` sections and
   rebuilt, binary-searching for the first section that breaks the build.

Every build runs in a disposable git worktree (see ``cli.core.parallel``), so
the truncated chapters in phase 2 never touch the files in your checkout.
Chapter builds pass ``--skip-validate``: the question here is whether a
chapter renders, and references to chapters outside an isolated build would
otherwise always fail validation.

Logs and artifacts land in ``books/_build/debug/<vol>/<fmt>/<run-id>/``.

Usage:
    ./binder/binder debug pdf --vol1                      # scan, then bisect failures
    ./binder/binder debug pdf --vol1 --parallel 4         # scan four chapters at a time
    ./binder/binder debug html --vol2 --chapter training  # bisect one chapter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.discovery import AmbiguousChapterError, format_volume_display_name
from ..core.parallel import BuildJob, BuildSession, JobResult, new_run_id, run_jobs
from ..core.workspace import take_snapshot

console = Console()

#: Location of ``section_splitter.py``, which parses a chapter into sections.
CONTENT_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "tools" / "scripts" / "content"

#: Flags for every debug build: rendering is the question, not validation.
CHAPTER_BUILD_ARGS = ("--skip-validate",)

# Quarto warnings worth surfacing even when a build succeeds, as
# (pattern, human-readable label) pairs.
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
    """Return the known Quarto warnings found in *output*, deduplicated, in order of appearance."""
    found: List[str] = []
    for pattern, label in _QUARTO_WARN_PATTERNS:
        for match in pattern.finditer(output):
            detail = match.group(1) if match.lastindex else ""
            message = f"{label}: {detail}" if detail else label
            if message not in found:
                found.append(message)
    return found


def _failure_excerpt(result: JobResult, lines: int = 20) -> str:
    """Return the failure note plus the last *lines* lines of a job's log."""
    text = result.log_text().strip()
    tail = "\n".join(text.splitlines()[-lines:]) if text else ""
    return f"{result.note}\n{tail}".strip()


def _print_result(result: JobResult, verbose: bool = False) -> None:
    """Print PASS, WARN, or FAIL for a finished build, with warnings and, if verbose, the log tail."""
    warnings = _extract_quarto_warnings(result.log_text())
    if not result.ok:
        console.print(f"[red]FAIL[/red] ({result.seconds:.1f}s)")
    elif warnings:
        console.print(f"[yellow]WARN[/yellow] ({result.seconds:.1f}s)")
    else:
        console.print(f"[green]PASS[/green] ({result.seconds:.1f}s)")
    for warning in warnings:
        console.print(f"         [yellow]⚠ {escape(warning)}[/yellow]")
    if verbose and not result.ok:
        for line in _failure_excerpt(result).splitlines()[-3:]:
            console.print(f"         [dim]{escape(line)}[/dim]")


class DebugCommand:
    """Two-phase build debugger for the book volumes."""

    def __init__(self, config_manager, chapter_discovery, verbose: bool = False):
        """Store the shared configuration and discovery objects.

        Args:
            config_manager: ``ConfigManager`` for the checkout being debugged.
            chapter_discovery: ``ChapterDiscovery`` used to list and resolve chapters.
            verbose: Print the tail of each failed build's log.
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.verbose = verbose
        self.book_dir = config_manager.book_dir
        self.repo_root = config_manager.book_dir.parent

    def debug_build(
        self,
        format_type: str,
        volume: str,
        chapter: Optional[str] = None,
        workers: int = 1,
        keep_workspaces: bool = False,
    ) -> bool:
        """Run the debugger for one volume and format.

        Args:
            format_type: ``pdf``, ``html``, or ``epub``.
            volume: Volume to debug, such as ``vol1``.
            chapter: Skip the chapter scan and bisect this chapter.
            workers: Chapter builds to run at once during the scan.
            keep_workspaces: Leave the build worktrees on disk for inspection.

        Returns:
            True once debugging has finished, whatever it found; False when it
            could not run.
        """
        console.print(Panel(
            f"[bold red]Build Debugger[/bold red]\n"
            f"[dim]{format_volume_display_name(volume)} / {format_type.upper()}[/dim]",
            border_style="red",
        ))
        run_dir = self.book_dir / "_build" / "debug" / volume / format_type / new_run_id()
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Debug logs: {run_dir}[/dim]")

        if chapter:
            console.print(f"\n[bold]Section-level debug for [cyan]{escape(chapter)}[/cyan][/bold]\n")
            return self._bisect_chapter(chapter, volume, format_type, run_dir, keep_workspaces)

        console.print("\n[bold]Phase 1:[/bold] Building each chapter on its own...\n")
        failures = self._scan_chapters(volume, format_type, run_dir, workers, keep_workspaces)
        if failures is None:
            return False
        if not failures:
            console.print(Panel(
                "[bold green]All chapters build successfully.[/bold green]\n"
                "[dim]No failures to debug.[/dim]",
                border_style="green",
            ))
            return True

        console.print(f"\n[bold red]Found {len(failures)} failing chapter(s):[/bold red]")
        for result in failures:
            console.print(f"  [red]x[/red] {escape(result.job.chapter or result.job.name)}")
        for index, result in enumerate(failures, 1):
            console.print(f"\n[bold]Phase 2 ({index}/{len(failures)}):[/bold] "
                          f"Section-level debug for [cyan]{escape(result.job.chapter)}[/cyan]\n")
            self._bisect_chapter(result.job.chapter, volume, format_type, run_dir, keep_workspaces)
        return True

    def _scan_chapters(
        self, volume: str, format_type: str, run_dir: Path, workers: int, keep_workspaces: bool
    ) -> Optional[List[JobResult]]:
        """Build every chapter of *volume* on its own and return the failed results.

        Returns ``None`` when the volume has no chapters to build.
        """
        chapters = self.chapter_discovery.get_chapters_from_config(volume)
        if not chapters:
            console.print("[red]No chapters found.[/red]")
            return None
        jobs = [BuildJob(format_type, volume, stem, CHAPTER_BUILD_ARGS, label=f"{index:02d}_{stem}")
                for index, stem in enumerate(chapters, 1)]
        console.print(f"[dim]{len(jobs)} chapters, {workers} build(s) at a time[/dim]\n")

        def report(result: JobResult) -> None:
            """Print a finished chapter's name followed by its build result."""
            console.print(f"  {escape(result.job.chapter):<32s}", end=" ")
            _print_result(result, self.verbose)

        results = run_jobs(self.repo_root, jobs, run_dir / "phase1", workers=workers,
                           keep_workspaces=keep_workspaces, on_finish=report)
        failures = [result for result in results if not result.ok]
        console.print(f"\n[dim]Results: {len(results) - len(failures)} passed, {len(failures)} failed[/dim]")
        return failures

    def _bisect_chapter(
        self, chapter_name: str, volume: str, format_type: str, run_dir: Path, keep_workspaces: bool
    ) -> bool:
        """Binary-search one chapter for the first section that breaks its build.

        Returns:
            True once the search has finished; False when the chapter cannot
            be found or parsed.
        """
        sys.path.insert(0, str(CONTENT_SCRIPTS_DIR))
        try:
            from section_splitter import split_chapter
        except ImportError:
            console.print("[red]Cannot import section_splitter.py from content scripts.[/red]")
            console.print(f"[dim]Expected at: {CONTENT_SCRIPTS_DIR / 'section_splitter.py'}[/dim]")
            return False

        spec = chapter_name if "/" in chapter_name else f"{volume}/{chapter_name}"
        try:
            qmd_path = self.chapter_discovery.find_chapter_file(spec, allow_fuzzy=True)
        except AmbiguousChapterError as error:
            console.print(f"[red]Ambiguous chapter {escape(chapter_name)}:[/red] "
                          f"{escape(', '.join(error.locations))}")
            return False
        if qmd_path is None:
            console.print(f"[red]Chapter not found: {escape(chapter_name)}[/red]")
            return False

        console.print(f"[dim]Parsing {qmd_path.name} into sections...[/dim]")
        chapter = split_chapter(str(qmd_path))
        if not chapter.sections:
            console.print("[yellow]No ## sections found in chapter.[/yellow]")
            return False

        section_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        section_table.add_column("#", style="dim", width=4)
        section_table.add_column("Section", width=50)
        section_table.add_column("Lines", style="dim", width=12)
        for section in chapter.sections:
            section_table.add_row(str(section.index), escape(section.title),
                                  f"L{section.start_line}-{section.end_line}")
        console.print(section_table)
        console.print()

        steps_dir = run_dir / "phase2" / qmd_path.stem
        relative = qmd_path.resolve().relative_to(self.repo_root.resolve())
        with BuildSession(take_snapshot(self.repo_root), steps_dir, name=f"bisect-{qmd_path.stem}",
                          run_id=new_run_id(), keep=keep_workspaces) as session:
            failing = self._binary_search_sections(session, chapter, qmd_path.stem, volume,
                                                   format_type, relative)

        console.print()
        if failing is None:
            console.print(Panel(
                f"[bold green]All {len(chapter.sections)} sections in {escape(qmd_path.stem)} "
                "build successfully.[/bold green]",
                border_style="green",
            ))
        elif failing == -1:
            console.print(Panel(
                "[bold red]Preamble itself fails to build.[/bold red]\n"
                "[dim]The issue is in the YAML frontmatter or content before the first ## section.[/dim]",
                border_style="red",
            ))
        else:
            section = chapter.sections[failing]
            text = Text()
            text.append("Build breaks at section ", style="bold red")
            text.append(f"{failing}", style="bold cyan")
            text.append(": ", style="bold red")
            text.append(f'"{section.title}"', style="bold white")
            text.append(f"\n\nFile:   {qmd_path.name}", style="dim")
            text.append(f"\nLines:  {section.start_line}-{section.end_line}", style="dim")
            if section.section_id:
                text.append(f"\nID:     #{section.section_id}", style="dim")
            text.append(f"\nLogs:   {steps_dir}", style="dim")
            console.print(Panel(text, title="Result", border_style="red"))
        return True

    def _binary_search_sections(
        self, session: BuildSession, chapter, stem: str, volume: str, format_type: str, relative: Path
    ) -> Optional[int]:
        """Find the first section whose inclusion breaks the build.

        Each step writes a truncated chapter into the session's workspace and
        builds it there.

        Returns:
            The failing section index, -1 if the preamble alone fails, or
            ``None`` if the whole chapter builds.
        """
        def builds(label: str, up_to_section: int) -> bool:
            """Build the chapter truncated after *up_to_section* and return whether it succeeded."""
            content = self._assemble_content(chapter, up_to_section)
            job = BuildJob(format_type, volume, stem, CHAPTER_BUILD_ARGS, label=label)
            result = session.run(
                job, prepare=lambda root: (root / relative).write_text(content, encoding="utf-8"))
            _print_result(result, self.verbose)
            return result.ok

        count = len(chapter.sections)
        console.print("  [dim]\\[pre][/dim]  Preamble only", end="  ")
        if not builds("preamble", -1):
            return -1
        console.print(f"  [dim]\\[full][/dim] All {count} sections", end="  ")
        if builds("full", count - 1):
            return None

        low, high, build_count = 0, count - 1, 2
        while low < high:
            middle = (low + high) // 2
            build_count += 1
            title = escape(chapter.sections[middle].title[:40])
            console.print(f'  [dim]\\[bisect {build_count}][/dim] Up to #{middle}: "{title}"', end="  ")
            if builds(f"step_{build_count:02d}_upto_{middle}", middle):
                low = middle + 1
            else:
                high = middle
        console.print(f"\n[dim]Isolated in {build_count} builds (binary search).[/dim]")
        return low

    @staticmethod
    def _assemble_content(chapter, up_to_section: int) -> str:
        """Return the chapter's frontmatter and preamble plus sections 0 through *up_to_section*.

        Args:
            chapter: ``ChapterStructure`` from ``section_splitter.split_chapter``.
            up_to_section: Highest section index to include; -1 keeps only the preamble.
        """
        parts = [part for part in (chapter.frontmatter, chapter.pre_content) if part]
        parts.extend(section.content for section in chapter.sections if section.index <= up_to_section)
        return "\n".join(parts)
