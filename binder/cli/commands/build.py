"""
Build command implementation for MLSysBook CLI.

Handles building chapters and full books in different formats (HTML, PDF, EPUB).
"""

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.discovery import format_volume_display_name
from ..core.process import (
    interrupts_as_keyboard_interrupt,
    local_render_env,
    start_process_group,
    stop_process_group,
)

console = Console()

#: Quarto ``--to`` target for each binder output format.
RENDER_TARGETS = {"html": "html", "pdf": "titlepage-pdf", "epub": "epub"}


def render_command(format_type: str) -> List[str]:
    """Return the ``quarto render`` command for *format_type*.

    Raises:
        ValueError: If the format is not html, pdf, or epub.
    """
    if format_type not in RENDER_TARGETS:
        raise ValueError(f"Unknown format type: {format_type}")
    return ["quarto", "render", f"--to={RENDER_TARGETS[format_type]}"]


class BuildCommand:
    """Handles build operations for the MLSysBook."""

    def __init__(self, config_manager, chapter_discovery, verbose: bool = False, open_after: bool = False):
        """Initialize build command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
            verbose: If True, stream build output in real-time
            open_after: If True, open build output after successful build
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.verbose = verbose
        self.open_after = open_after

    def _preflight_epub_hygiene(self, skip: bool = False) -> bool:
        """Run the `hygiene` scope of `./binder/binder check epub` before an EPUB build.

        Returns True if the build should proceed, False if it should abort.

        Rationale: the hygiene check runs in <1s across all SVG and
        BibTeX source files, while an EPUB render takes ~2 minutes per
        volume. Catching a regression now saves iterations later, and
        short-circuiting avoids producing a half-broken artifact that
        the user then has to diagnose.

        If hygiene fails, the method tells the user how to auto-repair
        (`./binder/binder check epub --scope hygiene --fix`) and how to bypass
        (`--skip-hygiene`) so the blocker is never mysterious.

        Opt-out: `--skip-hygiene` on the build command, or explicit
        `skip=True` for programmatic callers.
        """
        if skip:
            console.print("[yellow]⚠ Skipping EPUB hygiene preflight (--skip-hygiene)[/yellow]")
            return True

        # Lazy import: don't pay the cost when building HTML or PDF.
        try:
            from cli.commands._epub_checks import find_hygiene_issues
        except ImportError:
            # Running from an older checkout without the check module —
            # don't break the build, just note and continue.
            console.print("[dim]ⓘ Hygiene preflight unavailable in this checkout; skipping.[/dim]")
            return True

        repo_root = self.config_manager.root_dir
        issues, files_checked = find_hygiene_issues(repo_root)

        if not issues:
            console.print(
                f"[green]✓ EPUB hygiene preflight[/green] [dim]"
                f"({files_checked} SVG/BibTeX files scanned, 0 issues)[/dim]"
            )
            return True

        console.print(
            f"[red]✗ EPUB hygiene preflight failed[/red] [dim]"
            f"({len(issues)} issue(s) across {files_checked} scanned files)[/dim]"
        )
        # Show up to 5 issues so the user sees the shape of the problem
        # without drowning in output. Full detail is one command away.
        for issue in issues[:5]:
            console.print(
                f"  [red]{issue.code}[/red] "
                f"[dim]{issue.file}:{issue.line}[/dim] — {issue.message}"
            )
        if len(issues) > 5:
            console.print(f"  [dim]… {len(issues) - 5} more. "
                          f"See `./binder/binder check epub --scope hygiene` for all.[/dim]")
        console.print()
        console.print("[yellow]Resolve with one of:[/yellow]")
        console.print("  [cyan]./binder/binder check epub --scope hygiene --fix[/cyan]  "
                      "[dim]# auto-repair source files in place[/dim]")
        console.print("  [cyan]./binder/binder build epub --skip-hygiene[/cyan]       "
                      "[dim]# build anyway (not recommended)[/dim]")
        return False

    def _postflight_epub_validation(self, skip: bool = False, output_dir: Optional[Path] = None) -> bool:
        """Run smoke + epubcheck against the built EPUB(s) after render.

        Returns True if validation passed (or was skipped), False on a
        hard failure the caller should surface via a non-zero exit code.

        Called from every `./binder/binder build epub` entry point on success.
        Mirrors the CI gate so a local build that passes here will pass
        the `epub-validate` job in book-validate-dev.yml, and vice versa.

        Rationale: the pre-render hygiene preflight catches source-level
        problems, but it cannot catch renderer-emitted issues (markup
        the post-process sanitizer handled, or a new class epubcheck
        flags after a Quarto upgrade). Running smoke + epubcheck at
        the end closes that loop without asking the user to remember
        a follow-up command. Roughly ~7s added to a ~120s build.

        Opt-out: `--skip-validate` on the build command, or explicit
        `skip=True` for programmatic callers.
        """
        if skip:
            console.print("[yellow]⚠ Skipping post-build EPUB validation (--skip-validate)[/yellow]")
            return True

        try:
            from cli.commands._epub_checks import (
                _discover_built_epubs,
                emit_github_annotations,
                run_epubcheck_on,
                run_smoke_checks_on,
            )
        except ImportError:
            console.print("[dim]ⓘ Post-build validation unavailable in this checkout; skipping.[/dim]")
            return True

        repo_root = self.config_manager.root_dir
        epubs = sorted(output_dir.glob("*.epub")) if output_dir is not None else _discover_built_epubs(repo_root)
        if not epubs:
            console.print("[dim]ⓘ No EPUB artifacts found under _build/epub-vol*/; skipping post-build validation.[/dim]")
            return True

        console.print("[cyan]🔍 Post-build EPUB validation[/cyan]")
        overall_ok = True

        # --- Smoke (reader-compat, no Java) ----------------------------
        smoke_issues = []
        for epub in epubs:
            smoke_issues.extend(run_smoke_checks_on(epub, repo_root=repo_root))

        if smoke_issues:
            console.print(
                f"  [red]✗ smoke[/red]: {len(smoke_issues)} reader-compat issue(s)"
            )
            for issue in smoke_issues[:5]:
                console.print(
                    f"    [red]{issue.code}[/red] "
                    f"[dim]{issue.file}:{issue.line}[/dim] — {issue.message}"
                )
            if len(smoke_issues) > 5:
                console.print(f"    [dim]… {len(smoke_issues) - 5} more[/dim]")
            overall_ok = False
        else:
            console.print(f"  [green]✓ smoke[/green] [dim]({len(epubs)} EPUB(s), reader-compat clean)[/dim]")

        # --- Epubcheck (W3C, needs Java) -------------------------------
        # Use the same baseline file CI uses so local behaviour matches.
        baseline_path = repo_root / "binder" / "tools" / "audit" / "epubcheck-baseline.json"
        baseline_counts: Dict[str, Dict[str, int]] = {}
        if baseline_path.exists():
            try:
                import json as _json
                baseline_counts = _json.loads(
                    baseline_path.read_text(encoding="utf-8")
                ).get("volumes", {}) or {}
            except (OSError, ValueError):
                baseline_counts = {}

        total_fatal = 0
        total_errors = 0
        any_epubcheck_issue = False
        ratchet_violations: List[str] = []

        for epub in epubs:
            vol_key = epub.parent.name.replace("epub-", "") or epub.stem
            issues, counts = run_epubcheck_on(epub, repo_root=repo_root)
            emit_github_annotations(issues)
            total_fatal += counts.get("FATAL", 0)
            total_errors += counts.get("ERROR", 0)
            if issues:
                any_epubcheck_issue = True

            # Ratchet comparison per volume
            recorded = baseline_counts.get(vol_key, {})
            for severity in ("FATAL", "ERROR", "WARNING"):
                current = counts.get(severity, 0)
                allowed = int(recorded.get(severity, 0))
                if current > allowed:
                    ratchet_violations.append(
                        f"{vol_key} {severity}: {current} > baseline {allowed} "
                        f"(+{current - allowed})"
                    )

            # Missing-epubcheck sentinel — short-circuit gracefully.
            if any(i.code == "epubcheck-missing" for i in issues):
                console.print(
                    "  [yellow]⚠ epubcheck[/yellow]: not installed locally; "
                    "skipping full validation "
                    "[dim](install via `pip install epubcheck` or `./binder/binder doctor` for details)[/dim]"
                )
                # Don't fail the build just because Java is missing locally —
                # CI will catch anything this local pass would have.
                return overall_ok

        if ratchet_violations:
            console.print(
                f"  [red]✗ epubcheck[/red]: regression against baseline "
                f"({total_fatal} FATAL, {total_errors} ERROR total)"
            )
            for v in ratchet_violations:
                console.print(f"    [red]•[/red] {v}")
            console.print(
                "    [dim]If this is an accepted increase, rerun with "
                "`./binder/binder check epub --scope epubcheck --baseline "
                "binder/tools/audit/epubcheck-baseline.json --update-baseline` "
                "and commit the updated JSON.[/dim]"
            )
            overall_ok = False
        elif any_epubcheck_issue:
            # Issues exist but all within baseline — inform, don't fail.
            console.print(
                f"  [yellow]⚠ epubcheck[/yellow]: "
                f"{total_fatal} FATAL, {total_errors} ERROR "
                "[dim](all within baseline — no regression)[/dim]"
            )
        else:
            console.print(
                f"  [green]✓ epubcheck[/green] [dim]({len(epubs)} EPUB(s); "
                f"0 FATAL, 0 ERROR)[/dim]"
            )

        if not overall_ok:
            console.print()
            console.print(
                "[red]Build artifact written but validation failed.[/red] "
                "Inspect the issues above, then either fix the underlying "
                "cause or rebuild with [cyan]--skip-validate[/cyan] if "
                "you need the artifact as-is."
            )

        return overall_ok

    def _postflight_pdf_validation(self, volume: str, skip: bool = False, log_path=None) -> bool:
        """Scan the built volume PDF for unresolved refs and render leaks.

        Returns True if validation passed (or was skipped), False on findings
        the caller should surface via a non-zero exit code.

        Called from ``./binder/binder build pdf --vol1|--vol2`` on success. Quarto
        can exit 0 while leaving ``?@sec-foo`` literals or ``Figure ??`` in
        the PDF — this closes that loop without a separate manual check.

        Opt-out: ``--skip-validate`` on the build command.
        """
        if skip:
            console.print("[yellow]⚠ Skipping post-build PDF validation (--skip-validate)[/yellow]")
            return True

        vol_names = {
            "vol1": "Volume I",
            "vol2": "Volume II",
            "vol3": "Volume III",
            "vol4": "Volume IV",
            "tinytorch": "TinyTorch",
        }
        volume_name = vol_names.get(volume, volume)
        console.print(f"[cyan]🔍 Post-build PDF validation ({volume_name})[/cyan]")

        from cli.commands._pdf_checks import format_checklist, verify_volume_pdf

        quarto_dir = self.config_manager.book_dir
        result = verify_volume_pdf(quarto_dir, volume, log_path=log_path)
        console.print(format_checklist(result))

        # Rendered geometry detail. Physical trim crossings are release
        # blockers; text-box overflow and margin crowding remain triage rows.
        # The summary is computed inside verify_volume_pdf, so this reports the
        # same Binder-native check instead of the older pdfplumber heuristic.
        geom = getattr(result, "margin_geometry", None)
        if geom is not None and getattr(geom, "findings", None):
            console.print()
            edge_overflows = list(getattr(geom, "page_edge_overflows", []))
            if edge_overflows:
                console.print(
                    f"  [bold red]✗ RELEASE BLOCKER: {len(edge_overflows)} rendered "
                    f"object(s) cross the physical trim boundary[/bold red]"
                )
                for finding in edge_overflows[:10]:
                    console.print(
                        f"    [red]sheet {finding.page}: {finding.issue} "
                        f"{finding.side} — {finding.detail}[/red]"
                    )
            console.print(
                f"  [yellow]⚠ margin geometry[/yellow] [dim]("
                f"{len(geom.findings)} rendered margin finding(s) — "
                f"non-blocking in build validation)[/dim]"
            )
            edge_ids = {id(finding) for finding in edge_overflows}
            issue_rank = {
                "trim-overflow-bottom": 0,
                "trim-overflow-top": 0,
                "trim-overflow-left": 0,
                "trim-overflow-right": 0,
                "overlap": 1,
                "overflow-bottom": 2,
                "overflow-top": 3,
            }
            for finding in sorted(
                geom.findings,
                key=lambda f: (
                    0 if id(f) in edge_ids else 1,
                    issue_rank.get(f.issue, 9),
                    f.page,
                ),
            )[:10]:
                console.print(
                    f"    [dim]sheet {finding.page}: {finding.issue} "
                    f"{finding.side} — {finding.detail}[/dim]"
                )
            console.print(
                f"    [dim]→ `binder layout overlaps {result.pdf_path}` to localize; "
                f"`binder layout margins {result.pdf_path}` is the strict gate.[/dim]"
            )

        if not result.ok:
            console.print()
            console.print(
                "[red]Build artifact written but PDF validation failed.[/red] "
                "Inspect the issues above, fix the source QMD, and rebuild — or use "
                "[cyan]--skip-validate[/cyan] if you need the artifact as-is."
            )
            return False

        console.print(
            f"  [green]✓ pdf[/green] [dim]({result.pdf_path.name}: no unresolved refs or render errors in text)[/dim]"
        )
        return True

    def _open_output(self, output_dir: Path, format_type: str) -> None:
        """Open the build output using the system's default application.

        Uses shared rule: PDF = any .pdf, EPUB = any .epub, HTML = index.html.
        Always prints a clickable "Output created:" line for Cursor/VSCode terminals.
        """
        from ..core.config import get_output_file
        target = get_output_file(output_dir, format_type)

        if target is None:
            if self.open_after:
                console.print(f"[yellow]⚠️  No {format_type.upper()} output found to open in {output_dir}/[/yellow]")
            return

        # Print absolute path — Cursor/VSCode terminals auto-linkify file paths
        console.print(f"Output created: {target.resolve()}")

        if not self.open_after:
            return

        console.print(f"[cyan]🔗 Opening {target.name}...[/cyan]")

        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(target)])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", str(target)])
        elif system == "Windows":
            subprocess.Popen(["start", "", str(target)], shell=True)

    def build_full(self, format_type: str = "html", skip_hygiene: bool = False, skip_validate: bool = False) -> bool:
        """Build the whole book in one format with the default (Volume I) configuration.

        Args:
            format_type: Format to build ('html', 'pdf', 'epub')
            skip_hygiene: For EPUB builds, skip the pre-render hygiene
                check. Opt-in escape hatch for when a build must proceed
                despite source-level invariants (rare).
            skip_validate: For EPUB builds, skip post-render validation
                (epubcheck and smoke checks).

        Returns:
            True if build and post-build validation succeeded, False otherwise
        """
        console.print(f"[green]🔨 Building full {format_type.upper()} book...[/green]")
        console.print("[dim]📄 Building all files (full book mode)[/dim]")

        # EPUB preflight: catch source-level regressions before the
        # ~2-minute render rather than after.
        if format_type == "epub" and not self._preflight_epub_hygiene(skip=skip_hygiene):
            return False

        output_dir = self.config_manager.get_output_dir(format_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config_manager.activate_config(format_type)
        render_cmd = render_command(format_type)
        console.print(f"[blue]💻 Command: {' '.join(render_cmd)}[/blue]")

        try:
            # Ctrl-C unwinds through _run_command, which stops the renderer.
            with interrupts_as_keyboard_interrupt():
                success = self._run_command(
                    render_cmd,
                    cwd=self.config_manager.book_dir,
                    description=f"Building full {format_type.upper()} book",
                )
        except KeyboardInterrupt:
            console.print("[yellow]Build interrupted.[/yellow]")
            return False

        if not success:
            console.print(f"[red]❌ {format_type.upper()} build failed[/red]")
            return False
        console.print(f"[green]✅ {format_type.upper()} build completed: {output_dir}/[/green]")
        self._open_output(output_dir, format_type)
        # Post-flight validation closes the build→verify loop.
        if format_type == "epub":
            return self._postflight_epub_validation(skip=skip_validate, output_dir=output_dir)
        return True

    def build_chapters(self, chapter_names: List[str], format_type: str = "html", skip_hygiene: bool = False, skip_validate: bool = False) -> bool:
        """Build chapters named without a volume flag.

        The names are resolved, their volume is inferred, and the build is
        delegated to :meth:`build_chapters_with_volume`. A build renders one
        volume's configuration, so chapters from several volumes are rejected.

        Args:
            chapter_names: Chapter names, titles, paths, or patterns.
            format_type: Format to build ('html', 'pdf', 'epub')
            skip_hygiene: EPUB-only; skip the pre-render hygiene check.
            skip_validate: Skip post-render validation (EPUB smoke/epubcheck;
                PDF unresolved-ref scan).

        Returns:
            True if build and post-build validation succeeded, False otherwise
        """
        # Expand patterns like appendix* / re:^appendix_
        chapter_names = self.chapter_discovery.expand_chapter_patterns(chapter_names)
        try:
            resolved = self.chapter_discovery.validate_chapters(chapter_names)
        except Exception as error:
            console.print(f"[red]Build failed: {error}[/red]")
            return False

        volumes = {self.chapter_discovery._get_volume_from_path(path) for path in resolved}
        if len(volumes) != 1 or None in volumes:
            owners = ", ".join(sorted(volume or "shared" for volume in volumes))
            console.print(f"[red]These chapters do not belong to exactly one volume ({owners}).[/red]")
            console.print("[yellow]Build one volume at a time: "
                          "./binder/binder build <fmt> <chapters> --volN[/yellow]")
            return False
        return self.build_chapters_with_volume(
            [path.relative_to(self.config_manager.book_dir).as_posix() for path in resolved],
            format_type, next(iter(volumes)), skip_hygiene=skip_hygiene, skip_validate=skip_validate)

    def build_chapters_with_volume(self, chapter_names: List[str], format_type: str, volume: str, skip_hygiene: bool = False, skip_validate: bool = False) -> bool:
        """Build the volume index plus selected chapters into their own output directory.

        Output goes to ``<volume output>/chapters/<stem>[--<stem>...]``. Only the
        generated entry points (``_quarto.yml`` and ``index.qmd``) change during
        rendering; the canonical volume configs stay untouched, and the previous
        entry points are restored even if rendering fails or is interrupted
        (2026-09-11).

        Args:
            chapter_names: Chapter names, titles, paths, or patterns within *volume*.
            format_type: ``html``, ``pdf``, or ``epub``.
            volume: Volume whose configuration drives the build.
            skip_hygiene: EPUB-only; skip the pre-render hygiene check.
            skip_validate: Skip post-render validation.

        Returns:
            True if the build produced its artifact and validation passed.
        """
        import yaml
        from ..core.config import ACTIVE_CONFIG_MARKER, get_output_file
        from ..core.volume_index import volume_index_source
        from ..core.discovery import VOLUME_DIRS

        snapshots = {}
        try:
            if volume not in VOLUME_DIRS or format_type not in {"html", "pdf", "epub"}:
                raise ValueError(f"Unsupported volume/format: {volume}/{format_type}")
            chapter_names = self.chapter_discovery.expand_chapter_patterns(chapter_names, volume=volume)
            prefixed = []
            for name in chapter_names:
                specified, _ = self.chapter_discovery._parse_chapter_spec(name)
                if specified and specified != volume:
                    raise ValueError(f"Chapter {name!r} is outside selected {volume}")
                prefixed.append(name if specified else f"{volume}/{name}")
            chapter_files = list(dict.fromkeys(self.chapter_discovery.validate_chapters(prefixed)))
            if not chapter_files:
                raise ValueError("Select at least one chapter")

            # Do not fall back to another volume when this manifest is missing.
            books = self.config_manager.book_dir
            config_file = books / "config" / f"_quarto-{format_type}-{volume}.yml"
            config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            index_source = volume_index_source(books, volume, format_type)
            if not index_source.is_file():
                raise FileNotFoundError(f"Volume entry point not found: {index_source}")
            selected = [p.relative_to(books).as_posix() for p in chapter_files
                        if p.resolve() != index_source.resolve()]
            files = ["index.qmd", *selected]
            slug = "--".join(p.stem for p in chapter_files)
            output_dir = self.config_manager.get_output_dir(format_type, volume) / "chapters" / slug
            config.setdefault("project", {})["output-dir"] = output_dir.relative_to(books).as_posix()
            if format_type == "html":
                config["project"]["render"] = files
            else:
                config.setdefault("book", {})["chapters"] = files
                config["book"].pop("appendices", None)
                config["book"]["output-file"] = slug
                # The canonical volume config can carry a render list as well.
                config["project"].pop("render", None)

            if format_type == "epub" and not self._preflight_epub_hygiene(skip=skip_hygiene):
                return False
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in (self.config_manager.active_config, self.config_manager.active_index):
                snapshots[path] = (os.readlink(path), None) if path.is_symlink() else (
                    None, path.read_bytes() if path.exists() else None)

            with interrupts_as_keyboard_interrupt():
                self.config_manager.activate_config(format_type, volume)
                header = (f"{ACTIVE_CONFIG_MARKER}{config_file.relative_to(books).as_posix()}\n"
                          "# Temporary selective build; canonical config is unchanged.\n")
                self.config_manager.active_config.write_text(
                    header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
                console.print(f"[green]Building {volume} ({format_type}):[/green]")
                for path in files:
                    console.print(f"  {path}")
                success = self._run_command(
                    render_command(format_type), cwd=books,
                    description=f"Building {volume}: {', '.join(p.stem for p in chapter_files)}")
            if not success:
                return False
            artifact = get_output_file(output_dir, format_type)
            if artifact is None:
                console.print(f"[red]Build produced no {format_type} artifact in {output_dir}[/red]")
                return False
            console.print(f"[green]Build complete: {artifact}[/green]")
            self._open_output(output_dir, format_type)
            if format_type == "epub":
                return self._postflight_epub_validation(skip=skip_validate, output_dir=output_dir)
            if format_type == "pdf" and not skip_validate:
                from ._pdf_checks import verify_pdf, format_failure_report
                issues = verify_pdf(artifact, log_path=getattr(self, "_last_build_log", None))
                if issues:
                    console.print(format_failure_report("Chapter PDF", artifact, issues), markup=False)
                    console.print("[yellow]References to omitted chapters may be unresolved in this isolated build. "
                                  "Use --skip-validate for layout iteration; validate the full volume before release.[/yellow]")
                    return False
            return True
        except KeyboardInterrupt:
            console.print("[yellow]Build interrupted; restoring generated entry points.[/yellow]")
            return False
        except Exception as error:
            console.print(f"[red]Build failed: {error}[/red]")
            return False
        finally:
            for path, (link, content) in snapshots.items():
                if path.is_symlink():
                    path.unlink()
                if link is not None:
                    if path.exists():
                        path.unlink()
                    path.symlink_to(link)
                elif content is not None:
                    path.write_bytes(content)
                elif path.exists():
                    path.unlink()

    def build_volume(
        self,
        volume: str,
        format_type: str = "pdf",
        skip_hygiene: bool = False,
        skip_validate: bool = False,
        no_cover: bool = False,
        print_marks: bool = False,
    ) -> bool:
        """Build one whole volume using its dedicated configuration.

        Uses the volume-specific config (for example ``_quarto-pdf-vol1.yml``),
        which lists every chapter of that volume in order.

        Args:
            volume: Volume to build ('vol1' through 'vol4')
            format_type: Format to build ('html', 'pdf', 'epub')
            skip_hygiene: EPUB-only; skip the pre-render hygiene check.
            skip_validate: Skip post-render validation (EPUB smoke/epubcheck;
                PDF unresolved-ref scan).
            no_cover: PDF-only; omit the designed cover. The switch is set in
                the generated ``_quarto.yml``, so the source config never changes.
            print_marks: PDF-only; enable printer camera/trim marks in the
                shared TeX header for this build and restore it afterwards.

        Returns:
            True if build and post-build validation succeeded, False otherwise
        """
        volume_name = format_volume_display_name(volume)
        console.print(f"[magenta]📖 Building {volume_name} ({format_type.upper()})...[/magenta]")

        if format_type == "epub" and not self._preflight_epub_hygiene(skip=skip_hygiene):
            return False
        if (no_cover or print_marks) and format_type != "pdf":
            console.print("[red]❌ --no-cover and --print-marks apply to PDF builds only[/red]")
            return False

        config_file = self.config_manager.book_dir / "config" / f"_quarto-{format_type}-{volume}.yml"
        if not config_file.exists():
            console.print(f"[red]❌ Volume config not found: {config_file}[/red]")
            return False
        console.print(f"[dim]Using config: {config_file.name}[/dim]")

        output_dir = self.config_manager.get_output_dir(format_type, volume)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Every PDF includes the shared TeX header, so crop marks are switched
        # there for the duration of the build and restored afterwards.
        header_file = self.config_manager.book_dir / "shared" / "tex" / "header-includes.tex"
        header_original: Optional[str] = None

        try:
            with interrupts_as_keyboard_interrupt():
                config_name = self.config_manager.activate_config(format_type, volume)
                console.print(f"[dim]📄 Wrote _quarto.yml from {config_name}[/dim]")
                if no_cover:
                    if self._set_pdf_cover(self.config_manager.active_config, enabled=False):
                        console.print("[yellow]📘 Omitting the designed cover for this build[/yellow]")
                    else:
                        console.print("[dim]📘 Designed cover already omitted[/dim]")
                if print_marks:
                    header_original = header_file.read_text(encoding="utf-8")
                    if self._set_pdf_print_marks(header_file, enabled=True):
                        console.print("[yellow]✂ Temporarily enabled printer camera/trim marks[/yellow]")
                    else:
                        console.print("[dim]✂ Printer camera/trim marks already enabled[/dim]")
                render_cmd = render_command(format_type)
                console.print(f"[blue]💻 Command: {' '.join(render_cmd)}[/blue]")
                success = self._run_command(
                    render_cmd,
                    cwd=self.config_manager.book_dir,
                    description=f"Building {volume_name} ({format_type.upper()})",
                )
        except KeyboardInterrupt:
            console.print("[yellow]Build interrupted.[/yellow]")
            return False
        finally:
            if header_original is not None:
                header_file.write_text(header_original, encoding="utf-8")
                console.print("[green]✅ Print-mark setting restored[/green]")

        if not success:
            console.print(f"[red]❌ {volume_name} {format_type.upper()} build failed[/red]")
            return False
        console.print(f"[green]✅ {volume_name} {format_type.upper()} build completed: {output_dir}/[/green]")
        self._open_output(output_dir, format_type)
        if format_type == "epub":
            return self._postflight_epub_validation(skip=skip_validate, output_dir=output_dir)
        if format_type == "pdf":
            build_log = getattr(self, "_last_build_log", None)
            return self._postflight_pdf_validation(
                volume, skip=skip_validate,
                log_path=build_log if build_log and build_log.is_file() else None,
            )
        return True

    @staticmethod
    def _set_pdf_cover(config_file: Path, enabled: bool) -> bool:
        """Set the single titlepage PDF cover switch in a build manifest.

        ``build_volume`` applies it to the generated ``_quarto.yml`` copy, so the
        source configuration is never modified. Returning ``False`` means the
        switch was already in the requested state.
        """
        content = config_file.read_text(encoding="utf-8")
        pattern = re.compile(
            r"^(\s{4}coverpage:\s*)(false|true)(\s*(?:#.*)?)$",
            flags=re.MULTILINE,
        )
        matches = list(pattern.finditer(content))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one four-space `coverpage:` switch in "
                f"{config_file}; found {len(matches)}"
            )

        match = matches[0]
        desired = "true" if enabled else "false"
        if match.group(2) == desired:
            return False

        updated = pattern.sub(rf"\g<1>{desired}\g<3>", content, count=1)
        config_file.write_text(updated, encoding="utf-8")
        return True

    @staticmethod
    def _set_pdf_print_marks(header_file: Path, enabled: bool) -> bool:
        """Set the one active crop-mark switch in the shared TeX header."""
        content = header_file.read_text(encoding="utf-8")
        pattern = re.compile(
            r"^(\\CropMarks)(true|false)(\s*(?:%.*)?)$",
            flags=re.MULTILINE,
        )
        matches = list(pattern.finditer(content))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one active `\\CropMarks` switch in "
                f"{header_file}; found {len(matches)}"
            )

        desired = "true" if enabled else "false"
        match = matches[0]
        if match.group(2) == desired:
            return False

        updated = pattern.sub(rf"\g<1>{desired}\g<3>", content, count=1)
        header_file.write_text(updated, encoding="utf-8")
        return True

    def _run_command(self, cmd: List[str], cwd: Path, description: str) -> bool:
        """Run a command with progress indication.

        Args:
            cmd: Command to run
            cwd: Working directory
            description: Description for progress display

        Returns:
            True if command succeeded, False otherwise
        """
        env = local_render_env(self.config_manager.root_dir)
        process = None
        try:
            if self.verbose:
                # Verbose mode: stream output in real time and keep it for analysis.
                console.print(f"[dim]▶ {description}[/dim]")
                process = start_process_group(
                    cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                lines_buf: list[str] = []
                for line in iter(process.stdout.readline, ''):
                    if line:
                        console.print(line.rstrip())
                        lines_buf.append(line)
                process.wait(timeout=1800)
                self._save_build_log(cwd, "".join(lines_buf))
                if process.returncode != 0:
                    console.print(f"[red]Command failed with exit code {process.returncode}[/red]")
                return process.returncode == 0

            # Quiet mode: show a spinner and print diagnostics on failure.
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(description, total=None)
                process = start_process_group(
                    cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                stdout, stderr = process.communicate(timeout=1800)
                progress.update(task, completed=True)

            self._save_build_log(cwd, "".join(part for part in (stdout, stderr) if part))
            if process.returncode == 0:
                return True
            console.print(f"[red]Command failed with exit code {process.returncode}[/red]")
            if stderr and stderr.strip():
                console.print(f"[red]Error output (stderr):[/red]\n{stderr.strip()}")
            elif stdout and stdout.strip():
                # Quarto often writes fatal Lua/TeX/render errors to stdout.
                stdout_lines = stdout.strip().splitlines()
                error_lines = [
                    line for line in stdout_lines
                    if any(k in line.lower() for k in ("error", "fatal", "failed", "undefined control sequence", "compilation error"))
                ]
                if error_lines:
                    sample = "\n".join(error_lines[-15:])
                    console.print(f"[red]Error diagnostics detected in output:[/red]\n{sample}")
                else:
                    tail = "\n".join(stdout_lines[-20:])
                    console.print(f"[yellow]Last output lines before failure:[/yellow]\n{tail}")
            if cwd and getattr(self, "_last_build_log", None):
                console.print(f"[dim]Full build log saved at: {self._last_build_log}[/dim]")
            return False

        except subprocess.TimeoutExpired:
            stop_process_group(process)
            console.print("[red]❌ Build timed out after 30 minutes[/red]")
            return False
        except (KeyboardInterrupt, SystemExit):
            # Stop Quarto's process group (including TeX children) before the
            # caller restores shared generated files (2026-09-11).
            stop_process_group(process)
            raise
        except Exception as e:
            stop_process_group(process)
            console.print(f"[red]❌ Command execution error: {e}[/red]")
            return False

    def _save_build_log(self, cwd: Optional[Path], text: str) -> None:
        """Write renderer output to ``<cwd>/_build/last-build.log`` for post-build checks."""
        if cwd and text:
            build_log = Path(cwd) / "_build" / "last-build.log"
            build_log.parent.mkdir(parents=True, exist_ok=True)
            build_log.write_text(text, encoding="utf-8")
            self._last_build_log = build_log

    @staticmethod
    def _reset_config_comments(content: str) -> str:
        """Reset a config file by uncommenting all .qmd chapter lines.

        This ensures a clean starting state regardless of whether a previous
        build was interrupted and left the config in a partially-commented state.

        Handles patterns like:
            # - vol1/chapter/chapter.qmd  →  - vol1/chapter/chapter.qmd
            #- vol1/chapter/chapter.qmd   →  - vol1/chapter/chapter.qmd

        Also uncomments structural lines (part declarations, chapters: keys)
        that may have been commented out by a previous fast build.

        Args:
            content: Raw config file content

        Returns:
            Content with all .qmd lines and their structural containers uncommented
        """
        lines = content.split('\n')
        reset_lines = []

        for line in lines:
            stripped = line.strip()

            if (stripped.startswith('#') and '.qmd' in line and
                    (stripped.startswith('# - ') or stripped.startswith('#- '))):
                # Uncomment a previously-commented chapter list item.
                # The `# - ` / `#- ` prefix guard distinguishes list items
                # from prose comments that merely mention `.qmd` (which
                # must be left alone, otherwise we'd produce invalid YAML).
                indent = len(line) - len(line.lstrip())
                uncommented = stripped.lstrip('#').lstrip()
                if not uncommented.startswith('-'):
                    uncommented = '- ' + uncommented
                reset_lines.append(' ' * indent + uncommented)
            elif stripped.startswith('#') and (
                'part:' in stripped
                or stripped.lstrip('#').strip().startswith('chapters:')
                or stripped.lstrip('#').strip().startswith('appendices:')
            ):
                # Uncomment structural lines (part declarations, chapters/appendices keys)
                indent = len(line) - len(line.lstrip())
                uncommented = stripped.lstrip('#').lstrip()
                reset_lines.append(' ' * indent + uncommented)
            else:
                reset_lines.append(line)

        return '\n'.join(reset_lines)

    def reset_build_config(self, format_type: str, volume: Optional[str] = None) -> bool:
        """Reset build config by uncommenting all chapter entries.

        Restores YAML manifests that may have been partially commented by a
        chapter-selective build. PDF/EPUB use chapters:/appendices: lists;
        HTML uses render: lists where needed.

        Args:
            format_type: Format to reset ('html', 'pdf', 'epub')
            volume: Optional specific volume ('vol1' or 'vol2'). If omitted, reset both volumes.

        Returns:
            True if at least one config was reset successfully.
        """
        format_type = format_type.lower()
        if format_type not in {"html", "pdf", "epub"}:
            console.print(f"[red]❌ Unsupported format for reset: {format_type}[/red]")
            return False

        def _candidate_configs() -> List[Path]:
            """Return the config for ``volume``, or the Volume I and II configs when unset."""
            if volume:
                return [self.config_manager.get_config_file(format_type, volume)]

            # Reset all known volume configs for the selected format.
            if format_type == "html":
                return [self.config_manager.html_vol1_config, self.config_manager.html_vol2_config]
            if format_type == "pdf":
                return [self.config_manager.pdf_vol1_config, self.config_manager.pdf_vol2_config]
            return [self.config_manager.epub_vol1_config, self.config_manager.epub_vol2_config]

        targets = _candidate_configs()
        reset_count = 0

        for cfg in targets:
            if not cfg.exists():
                console.print(f"[yellow]⚠️ Skipping missing config: {cfg.name}[/yellow]")
                continue

            try:
                original_content = cfg.read_text(encoding="utf-8")
                reset_content = self._reset_config_comments(original_content)

                # All formats: uncomment any chapter/appendix/render entries a
                # chapter-selective build commented out, restoring the full
                # default manifest. For HTML this restores the render: list
                # (vol2) and is a no-op for the render-all volume (vol1). The
                # render: section is the HTML scoping mechanism and is no longer
                # stripped on reset — doing so would delete vol2's full manifest.
                cfg.write_text(reset_content, encoding="utf-8")

                backup_file = cfg.with_suffix(".backup")
                if backup_file.exists():
                    backup_file.unlink()

                reset_count += 1
                try:
                    rel = cfg.relative_to(self.config_manager.book_dir)
                except ValueError:
                    rel = cfg
                console.print(f"[green]✓[/green] Reset config: {rel}")
            except Exception as e:
                console.print(f"[red]❌ Failed to reset {cfg.name}: {e}[/red]")

        if reset_count == 0:
            console.print("[red]❌ No configs were reset.[/red]")
            return False

        scope = volume if volume else "all volumes"
        console.print(f"[green]✅ Reset {format_type.upper()} build config for {scope}[/green]")
        return True
