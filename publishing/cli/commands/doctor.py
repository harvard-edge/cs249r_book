"""
Health check command for MLSysBook CLI.

Performs comprehensive system health checks to ensure everything is working properly.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

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
            ("Python", [sys.executable, "--version"]),
            ("Quarto", ["quarto", "--version"]),
            ("Git", ["git", "--version"]),
            ("Node.js", ["node", "--version"]),
            ("R", ["R", "--version"]),
        ]

        for name, cmd in dependencies:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    self.checks.append({
                        "category": "Dependencies",
                        "name": name,
                        "passed": True,
                        "message": f"✅ {version}",
                        "details": None
                    })
                else:
                    self.checks.append({
                        "category": "Dependencies",
                        "name": name,
                        "passed": False,
                        "message": "❌ Not found or error",
                        "details": result.stderr.strip() if result.stderr else "Command failed"
                    })
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.checks.append({
                    "category": "Dependencies",
                    "name": name,
                    "passed": False,
                    "message": "❌ Not installed",
                    "details": f"Command '{' '.join(cmd)}' not found"
                })

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
                    required_sections = ["project", "book"]
                    missing_sections = [s for s in required_sections if s not in config_data]

                    if missing_sections:
                        self.checks.append({
                            "category": "Configuration",
                            "name": name,
                            "passed": False,
                            "message": "⚠️ Missing sections",
                            "details": f"Missing: {', '.join(missing_sections)}"
                        })
                    else:
                        self.checks.append({
                            "category": "Configuration",
                            "name": name,
                            "passed": True,
                            "message": "✅ Valid",
                            "details": None
                        })

                except Exception as e:
                    self.checks.append({
                        "category": "Configuration",
                        "name": name,
                        "passed": False,
                        "message": "❌ Parse error",
                        "details": str(e)
                    })
            else:
                self.checks.append({
                    "category": "Configuration",
                    "name": name,
                    "passed": False,
                    "message": "❌ Not found",
                    "details": f"File not found: {config_path}"
                })

        # Check active config symlink
        if self.config_manager.active_config.is_symlink():
            target = self.config_manager.active_config.readlink()
            self.checks.append({
                "category": "Configuration",
                "name": "Active Config",
                "passed": True,
                "message": f"✅ Linked to {target}",
                "details": None
            })
        elif self.config_manager.active_config.exists():
            self.checks.append({
                "category": "Configuration",
                "name": "Active Config",
                "passed": False,
                "message": "⚠️ Regular file (not symlink)",
                "details": "Should be a symlink to format-specific config"
            })
        else:
            self.checks.append({
                "category": "Configuration",
                "name": "Active Config",
                "passed": False,
                "message": "❌ Not found",
                "details": "No _quarto.yml found"
            })

    def _check_chapter_files(self) -> None:
        """Check chapter files and structure."""
        chapters = self.chapter_discovery.get_all_chapters()

        self.checks.append({
            "category": "Content",
            "name": "Chapter Count",
            "passed": len(chapters) > 0,
            "message": f"✅ {len(chapters)} chapters found" if len(chapters) > 0 else "❌ No chapters found",
            "details": None
        })

        # Check for common issues
        large_chapters = [ch for ch in chapters if ch["size"] > 500 * 1024]  # > 500KB
        if large_chapters:
            self.checks.append({
                "category": "Content",
                "name": "Large Chapters",
                "passed": False,
                "message": f"⚠️ {len(large_chapters)} large chapters (>500KB)",
                "details": f"Large chapters: {', '.join([ch['name'] for ch in large_chapters[:3]])}"
            })

        # Check for missing chapters (common ones).
        # Use discovered chapter names directly so duplicate names across volumes
        # (e.g., vol1/introduction and vol2/introduction) are not treated as errors.
        expected_chapters = ["introduction", "ml_systems", "training", "ops"]
        discovered_names = {ch["name"] for ch in chapters}
        missing_chapters = [expected for expected in expected_chapters if expected not in discovered_names]

        if missing_chapters:
            self.checks.append({
                "category": "Content",
                "name": "Core Chapters",
                "passed": False,
                "message": f"⚠️ {len(missing_chapters)} core chapters missing",
                "details": f"Missing: {', '.join(missing_chapters)}"
            })
        else:
            self.checks.append({
                "category": "Content",
                "name": "Core Chapters",
                "passed": True,
                "message": "✅ All core chapters present",
                "details": None
            })

    def _check_epub_health(self) -> None:
        """Check EPUB tooling availability and source hygiene.

        Emits four checks under the "EPUB" category:

          1. Java runtime (JRE 8+) — required by epubcheck.
          2. epubcheck (Python wrapper or system binary) — the W3C
             validator used by the `epubcheck` scope.
          3. Built EPUB artifacts — whether there is something for
             `./binder check epub --scope epubcheck` to validate.
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
                version_line = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "unknown"
                self.checks.append({
                    "category": "EPUB",
                    "name": "Java runtime",
                    "passed": True,
                    "message": f"✅ {version_line}",
                    "details": None,
                })
            except Exception as e:
                self.checks.append({
                    "category": "EPUB",
                    "name": "Java runtime",
                    "passed": False,
                    "message": "⚠️ `java` found but reported an error",
                    "details": str(e),
                })
        else:
            self.checks.append({
                "category": "EPUB",
                "name": "Java runtime",
                "passed": False,
                "message": "⚠️ `java` not on PATH",
                "details": (
                    "Required for epubcheck. Install with "
                    "`brew install temurin` (macOS), "
                    "`apt install default-jre` (Debian/Ubuntu), "
                    "or `dnf install java-17-openjdk` (Fedora)."
                ),
            })

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
            self.checks.append({
                "category": "EPUB",
                "name": "epubcheck",
                "passed": True,
                "message": f"✅ Available via {' + '.join(modes)}",
                "details": None,
            })
        else:
            self.checks.append({
                "category": "EPUB",
                "name": "epubcheck",
                "passed": False,
                "message": "⚠️ epubcheck not found",
                "details": (
                    "Install with `pip install epubcheck` (Python wrapper, "
                    "requires JRE 8+) or via your OS package manager "
                    "(`brew install epubcheck` / `apt install epubcheck`)."
                ),
            })

        # --- 3. Built EPUB artifacts ------------------------------------
        try:
            from cli.commands._epub_checks import _discover_built_epubs
        except ImportError:
            _discover_built_epubs = None

        # book_dir is `book/quarto/`; repo root is two levels up.
        repo_root = self.config_manager.book_dir.parent.parent
        if _discover_built_epubs is not None:
            epubs = _discover_built_epubs(repo_root)
            if epubs:
                names = ", ".join(p.name for p in epubs)
                self.checks.append({
                    "category": "EPUB",
                    "name": "Built artifacts",
                    "passed": True,
                    "message": f"✅ {len(epubs)} EPUB(s) under _build/epub-vol*/",
                    "details": names,
                })
            else:
                self.checks.append({
                    "category": "EPUB",
                    "name": "Built artifacts",
                    "passed": True,  # not a failure — just informational
                    "message": "📁 No built EPUBs yet — run `./binder build epub --vol1`",
                    "details": None,
                })
        else:
            self.checks.append({
                "category": "EPUB",
                "name": "Built artifacts",
                "passed": True,
                "message": "ⓘ _epub_checks module unavailable (older checkout)",
                "details": None,
            })

        # --- 4. Source hygiene ------------------------------------------
        try:
            from cli.commands._epub_checks import find_hygiene_issues
        except ImportError:
            self.checks.append({
                "category": "EPUB",
                "name": "Source hygiene",
                "passed": True,
                "message": "ⓘ hygiene check unavailable (older checkout)",
                "details": None,
            })
            return

        issues, files_checked = find_hygiene_issues(repo_root)
        if not issues:
            self.checks.append({
                "category": "EPUB",
                "name": "Source hygiene",
                "passed": True,
                "message": f"✅ clean ({files_checked} SVG/BibTeX files)",
                "details": None,
            })
        else:
            # Group by code for a compact summary.
            by_code: Dict[str, int] = {}
            for i in issues:
                by_code[i.code] = by_code.get(i.code, 0) + 1
            summary = ", ".join(f"{code}:{n}" for code, n in sorted(by_code.items()))
            self.checks.append({
                "category": "EPUB",
                "name": "Source hygiene",
                "passed": False,
                "message": f"❌ {len(issues)} issue(s) across {files_checked} files",
                "details": (
                    f"{summary}. Fix with "
                    "`./binder check epub --scope hygiene --fix`."
                ),
            })

    def _check_build_artifacts(self) -> None:
        """Check for build artifacts and output directories."""
        formats = ["html", "pdf", "epub"]

        for format_type in formats:
            output_dir = self.config_manager.get_output_dir(format_type)

            if output_dir.exists():
                # Count files in output directory
                files = list(output_dir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])

                self.checks.append({
                    "category": "Build Artifacts",
                    "name": f"{format_type.upper()} Output",
                    "passed": True,
                    "message": f"✅ {file_count} files in {output_dir.name}/",
                    "details": None
                })
            else:
                self.checks.append({
                    "category": "Build Artifacts",
                    "name": f"{format_type.upper()} Output",
                    "passed": True,  # Not an error if no builds exist yet
                    "message": "📁 No build artifacts (clean)",
                    "details": None
                })

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
                self.checks.append({
                    "category": "Git",
                    "name": "Repository",
                    "passed": False,
                    "message": "❌ Not a git repository",
                    "details": None
                })
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
                self.checks.append({
                    "category": "Git",
                    "name": "Current Branch",
                    "passed": True,
                    "message": f"✅ {branch}",
                    "details": None
                })

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
                    self.checks.append({
                        "category": "Git",
                        "name": "Working Tree",
                        "passed": True,  # Not necessarily an error
                        "message": f"📝 {change_count} uncommitted changes",
                        "details": None
                    })
                else:
                    self.checks.append({
                        "category": "Git",
                        "name": "Working Tree",
                        "passed": True,
                        "message": "✅ Clean",
                        "details": None
                    })

        except Exception as e:
            self.checks.append({
                "category": "Git",
                "name": "Status Check",
                "passed": False,
                "message": "❌ Error checking git status",
                "details": str(e)
            })

    def _check_permissions(self) -> None:
        """Check file permissions for key files."""
        key_files = [
            ("binder", self.config_manager.root_dir / "binder"),
        ]

        for name, file_path in key_files:
            if file_path.exists():
                is_executable = file_path.stat().st_mode & 0o111
                self.checks.append({
                    "category": "Permissions",
                    "name": name,
                    "passed": bool(is_executable),
                    "message": "✅ Executable" if is_executable else "❌ Not executable",
                    "details": f"chmod +x {file_path.name}" if not is_executable else None
                })
            else:
                self.checks.append({
                    "category": "Permissions",
                    "name": name,
                    "passed": False,
                    "message": "❌ File not found",
                    "details": None
                })

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
                f"[red]{failed_checks} issues found that may need attention.[/red]\n"
                "[dim]Review the details above and fix any critical issues.[/dim]",
                title="⚠️ Health Check Summary",
                border_style="yellow"
            ))
