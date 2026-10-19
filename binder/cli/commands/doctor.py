"""
Health check command for MLSysBook CLI.

Performs comprehensive system health checks to ensure everything is working properly.
"""

import subprocess
import sys
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.discovery import AmbiguousChapterError, discover_volumes

console = Console()


class DoctorCommand:
    """Performs health checks for the MLSysBook system."""

    def __init__(self, config_manager, chapter_discovery):
        """Initialize doctor command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.checks = []

    def _add(self, category: str, name: str, passed: bool, message: str, details: Optional[str] = None) -> None:
        """Record one check result for the summary tables."""
        self.checks.append({
            "category": category,
            "name": name,
            "passed": passed,
            "message": message,
            "details": details,
        })

    def run_health_check(self) -> bool:
        """Run comprehensive health check.

        Returns:
            True if all checks pass, False if any issues found
        """
        console.print("[bold blue]🏥 MLSysBook Health Check[/bold blue]")
        console.print("[dim]Running comprehensive system diagnostics...[/dim]\n")

        self.checks = []

        # Run all health checks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:

            # System dependencies
            task = progress.add_task("Checking system dependencies...", total=None)
            self._check_system_dependencies()
            progress.update(task, completed=True)

            # Configuration files
            task = progress.add_task("Validating configuration files...", total=None)
            self._check_configuration_files()
            progress.update(task, completed=True)

            # Chapter files
            task = progress.add_task("Scanning chapter files...", total=None)
            self._check_chapter_files()
            progress.update(task, completed=True)

            # Build artifacts
            task = progress.add_task("Checking build artifacts...", total=None)
            self._check_build_artifacts()
            progress.update(task, completed=True)

            # Git status
            task = progress.add_task("Checking git status...", total=None)
            self._check_git_status()
            progress.update(task, completed=True)

            # Permissions
            task = progress.add_task("Checking file permissions...", total=None)
            self._check_permissions()
            progress.update(task, completed=True)

            # EPUB tooling + source hygiene
            task = progress.add_task("Checking EPUB tooling and source hygiene...", total=None)
            self._check_epub_health()
            progress.update(task, completed=True)

        # Display results
        self._display_results()

        # Return overall health status
        failed_checks = [check for check in self.checks if not check["passed"]]
        return len(failed_checks) == 0

    def _check_system_dependencies(self) -> None:
        """Check required system dependencies."""
        dependencies = [
            ("Python", [sys.executable, "--version"], True),
            ("Quarto", ["quarto", "--version"], True),
            ("Git", ["git", "--version"], True),
            ("Node.js", ["node", "--version"], False),
        ]

        for name, cmd, required in dependencies:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    self._add("Dependencies", name, True, f"✅ {version}")
                else:
                    self._add(
                        "Dependencies",
                        name,
                        not required,
                        "⚠️ Optional tool error" if not required else "❌ Not found or error",
                        result.stderr.strip() if result.stderr else "Command failed",
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._add(
                    "Dependencies",
                    name,
                    not required,
                    "⚠️ Not installed (optional)" if not required else "❌ Not installed",
                    f"Command '{' '.join(cmd)}' not found",
                )

    def _check_configuration_files(self) -> None:
        """Check Quarto configuration files."""
        configs = [
            ("HTML Config", self.config_manager.html_config),
            ("PDF Config", self.config_manager.pdf_config),
            ("EPUB Config", self.config_manager.epub_config),
        ]

        for name, config_path in configs:
            if config_path.exists():
                try:
                    # Try to read and parse the config
                    format_type = name.lower().split()[0]
                    config_data = self.config_manager.read_config(format_type)

                    # Check for required sections
                    if format_type == "html":
                        # HTML can be configured as a website or a book
                        missing_sections = []
                        if "project" not in config_data:
                            missing_sections.append("project")
                        if "book" not in config_data and "website" not in config_data:
                            missing_sections.append("book or website")
                    else:
                        required_sections = ["project", "book"]
                        missing_sections = [s for s in required_sections if s not in config_data]

                    if missing_sections:
                        self._add(
                            "Configuration",
                            name,
                            False,
                            "⚠️ Missing sections",
                            f"Missing: {', '.join(missing_sections)}",
                        )
                    else:
                        self._add("Configuration", name, True, "✅ Valid")

                except Exception as e:
                    self._add("Configuration", name, False, "❌ Parse error", str(e))
            else:
                self._add("Configuration", name, False, "❌ Not found", f"File not found: {config_path}")

        # Check the generated active config
        source = self.config_manager.active_config_source()
        if source:
            self._add("Configuration", "Active Config", True, f"✅ Copied from {source}")
        elif self.config_manager.active_config.exists():
            self._add(
                "Configuration",
                "Active Config",
                False,
                "⚠️ Not generated by binder",
                "Run ./binder/binder switch <format> to regenerate _quarto.yml",
            )
        else:
            # Every build writes _quarto.yml from a source config, so a fresh
            # or cleaned checkout without one is healthy.
            self._add(
                "Configuration",
                "Active Config",
                True,
                "📁 None yet",
                "Each build writes _quarto.yml; ./binder/binder switch <format> writes one now",
            )

    def _check_chapter_files(self) -> None:
        """Check chapter files and structure."""
        chapters = self.chapter_discovery.get_all_chapters()

        self._add(
            "Content",
            "Chapter Count",
            len(chapters) > 0,
            f"✅ {len(chapters)} chapters found" if len(chapters) > 0 else "❌ No chapters found",
        )

        # Check for common issues
        large_chapters = [ch for ch in chapters if ch["size"] > 500 * 1024]  # > 500KB
        if large_chapters:
            self._add(
                "Content",
                "Large Chapters",
                False,
                f"⚠️ {len(large_chapters)} large chapters (>500KB)",
                f"Large chapters: {', '.join([ch['name'] for ch in large_chapters[:3]])}",
            )

        # Every chapter a volume's PDF config lists must resolve to a source
        # file. (A hard-coded list of "core" chapter names went stale when
        # chapter directories gained numeric prefixes and always failed.)
        missing_chapters = []
        for volume in discover_volumes(self.config_manager.book_dir):
            for stem in self.chapter_discovery.get_chapters_from_config(volume):
                try:
                    found = self.chapter_discovery.find_chapter_file(f"{volume}/{stem}")
                except AmbiguousChapterError:
                    found = True  # resolves, just not uniquely; not a missing chapter
                if not found:
                    missing_chapters.append(f"{volume}/{stem}")

        if missing_chapters:
            self._add(
                "Content",
                "Configured Chapters",
                False,
                f"❌ {len(missing_chapters)} configured chapter(s) have no source file",
                f"Missing: {', '.join(missing_chapters[:5])}"
                + (f" (+{len(missing_chapters) - 5} more)" if len(missing_chapters) > 5 else "")
                + "; fix books/config/_quarto-pdf-<vol>.yml or restore the file",
            )
        else:
            self._add("Content", "Configured Chapters", True, "✅ Every configured chapter has a source file")

    def _check_epub_health(self) -> None:
        """Check EPUB tooling availability and source hygiene.

        Emits four checks under the "EPUB" category:

          1. Java runtime (JRE 8+) — required by epubcheck.
          2. epubcheck (Python wrapper or system binary) — the W3C
             validator used by the `epubcheck` scope.
          3. Built EPUB artifacts — whether there is something for
             `./binder/binder check epub --scope epubcheck` to validate.
          4. Source hygiene — runs the hygiene scope in-process; any
             issue here is a ship-stopper for the next EPUB build.
        """
        import shutil
        import subprocess

        # --- 1. Java runtime --------------------------------------------
        java = shutil.which("java")
        if java:
            try:
                # java -version writes to stderr in every JDK distribution.
                out = subprocess.run(
                    [java, "-version"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    version_line = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "unknown"
                    self._add("EPUB", "Java runtime", True, f"✅ {version_line}")
                else:
                    err_msg = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "Command failed"
                    self._add("EPUB", "Java runtime", False, "⚠️ Java runtime not available", err_msg)
            except Exception as e:
                self._add("EPUB", "Java runtime", False, "⚠️ `java` found but reported an error", str(e))
        else:
            self._add(
                "EPUB",
                "Java runtime",
                False,
                "⚠️ `java` not on PATH",
                "Required for epubcheck. Install with "
                    "`brew install temurin` (macOS), "
                    "`apt install default-jre` (Debian/Ubuntu), "
                    "or `dnf install java-17-openjdk` (Fedora).",
            )

        # --- 2. epubcheck ------------------------------------------------
        epubcheck_cli = shutil.which("epubcheck")
        epubcheck_py = False
        try:
            import epubcheck  # noqa: F401
            epubcheck_py = True
        except ImportError:
            pass

        if epubcheck_cli or epubcheck_py:
            modes = []
            if epubcheck_cli:
                modes.append("system binary")
            if epubcheck_py:
                modes.append("Python wrapper")
            self._add("EPUB", "epubcheck", True, f"✅ Available via {' + '.join(modes)}")
        else:
            self._add(
                "EPUB",
                "epubcheck",
                False,
                "⚠️ epubcheck not found",
                "Install with `pip install epubcheck` (Python wrapper, "
                    "requires JRE 8+) or via your OS package manager "
                    "(`brew install epubcheck` / `apt install epubcheck`).",
            )

        # --- 3. Built EPUB artifacts ------------------------------------
        from ._epub_checks import _discover_built_epubs, find_hygiene_issues

        repo_root = self.config_manager.root_dir
        epubs = _discover_built_epubs(repo_root)
        if epubs:
            self._add(
                "EPUB",
                "Built artifacts",
                True,
                f"✅ {len(epubs)} EPUB(s) under _build/epub-vol*/",
                ", ".join(p.name for p in epubs),
            )
        else:
            self._add(
                "EPUB",
                "Built artifacts",
                True,
                "📁 No built EPUBs yet — run `./binder/binder build epub --vol1`",
            )

        # --- 4. Source hygiene ------------------------------------------
        issues, files_checked = find_hygiene_issues(repo_root)
        if not issues:
            self._add("EPUB", "Source hygiene", True, f"✅ clean ({files_checked} SVG/BibTeX files)")
        else:
            # Group by code for a compact summary.
            by_code: Dict[str, int] = {}
            for i in issues:
                by_code[i.code] = by_code.get(i.code, 0) + 1
            summary = ", ".join(f"{code}:{n}" for code, n in sorted(by_code.items()))
            self._add(
                "EPUB",
                "Source hygiene",
                False,
                f"❌ {len(issues)} issue(s) across {files_checked} files",
                f"{summary}. Fix with "
                    "`./binder/binder check epub --scope hygiene --fix`.",
            )

    def _check_build_artifacts(self) -> None:
        """Check for build artifacts and output directories."""
        formats = ["html", "pdf", "epub"]

        for format_type in formats:
            output_dir = self.config_manager.get_output_dir(format_type)

            if output_dir.exists():
                # Count files in output directory
                files = list(output_dir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])

                self._add(
                    "Build Artifacts",
                    f"{format_type.upper()} Output",
                    True,
                    f"✅ {file_count} files in {output_dir.name}/",
                )
            else:
                self._add(
                    "Build Artifacts",
                    f"{format_type.upper()} Output",
                    True,
                    "📁 No build artifacts (clean)",
                )

    def _check_git_status(self) -> None:
        """Check git repository status."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                cwd=self.config_manager.root_dir
            )

            if result.returncode != 0:
                self._add("Git", "Repository", False, "❌ Not a git repository")
                return

            # Check current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                cwd=self.config_manager.root_dir
            )

            if result.returncode == 0:
                branch = result.stdout.strip()
                self._add("Git", "Current Branch", True, f"✅ {branch}")

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.config_manager.root_dir
            )

            if result.returncode == 0:
                changes = result.stdout.strip()
                if changes:
                    change_count = len(changes.split('\n'))
                    self._add("Git", "Working Tree", True, f"📝 {change_count} uncommitted changes")
                else:
                    self._add("Git", "Working Tree", True, "✅ Clean")

        except Exception as e:
            self._add("Git", "Status Check", False, "❌ Error checking git status", str(e))

    def _check_permissions(self) -> None:
        """Check file permissions for key files."""
        key_files = [
            ("binder", self.config_manager.root_dir / "binder" / "binder"),
        ]

        for name, file_path in key_files:
            if file_path.exists():
                is_executable = file_path.stat().st_mode & 0o111
                self._add(
                    "Permissions",
                    name,
                    bool(is_executable),
                    "✅ Executable" if is_executable else "❌ Not executable",
                    f"chmod +x {file_path.name}" if not is_executable else None,
                )
            else:
                self._add("Permissions", name, False, "❌ File not found")

    def _display_results(self) -> None:
        """Display health check results in a formatted table."""
        console.print()

        # Group checks by category
        categories = {}
        for check in self.checks:
            cat = check["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(check)

        # Display each category
        for category, checks in categories.items():
            table = Table(show_header=True, header_style="bold blue", box=None)
            table.add_column("Check", style="white", width=20)
            table.add_column("Status", style="white", width=30)
            table.add_column("Details", style="dim", width=40)

            for check in checks:
                details = check["details"] if check["details"] else ""
                table.add_row(
                    check["name"],
                    check["message"],
                    details
                )

            console.print(Panel(table, title=f"🔍 {category}", border_style="cyan"))

        # Summary
        total_checks = len(self.checks)
        passed_checks = len([c for c in self.checks if c["passed"]])
        failed_checks = total_checks - passed_checks

        if failed_checks == 0:
            console.print(Panel(
                f"[bold green]🎉 All {total_checks} health checks passed![/bold green]\n"
                "[dim]Your MLSysBook environment is healthy and ready to use.[/dim]",
                title="✅ Health Check Summary",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold yellow]⚠️ {passed_checks}/{total_checks} checks passed[/bold yellow]\n"
                f"[red]{failed_checks} issues found that may need attention:[/red] "
                + ", ".join(f"{c['category']} / {c['name']}" for c in self.checks if not c["passed"])
                + "\n[dim]The Details column above says what to fix.[/dim]",
                title="⚠️ Health Check Summary",
                border_style="yellow"
            ))
