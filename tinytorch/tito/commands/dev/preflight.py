"""
Preflight checks for TinyTorch development and releases.

This command runs comprehensive verification before commits, PRs, or releases.
The same checks can be used in CI/CD pipelines.

Usage:
    tito dev preflight              # Standard preflight (quick + structure)
    tito dev preflight --full       # Full validation (includes module tests)
    tito dev preflight --release    # Release validation (comprehensive)
    tito dev preflight --ci         # CI mode (non-interactive, exit codes)
"""

import subprocess
import sys
import time
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Group
from rich.text import Text
from rich.live import Live
from rich.status import Status
from rich import box

from ..base import BaseCommand


class CheckStatus(Enum):
    """Status of a preflight check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    status: CheckStatus
    message: str = ""
    duration_ms: int = 0
    details: List[str] = field(default_factory=list)
    command: str = ""  # The command that was run
    stdout: str = ""   # Captured stdout
    stderr: str = ""   # Captured stderr


@dataclass
class CheckCategory:
    """A category of preflight checks."""
    name: str
    emoji: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class PreflightCommand(BaseCommand):
    """Run preflight checks before commits, PRs, or releases."""

    @property
    def name(self) -> str:
        return "preflight"

    @property
    def description(self) -> str:
        return "Run preflight verification checks"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--quick',
            action='store_true',
            help='Quick checks only (~10 seconds)'
        )
        parser.add_argument(
            '--full',
            action='store_true',
            help='Full validation including module tests (~2-5 minutes)'
        )
        parser.add_argument(
            '--release',
            action='store_true',
            help='Release validation - comprehensive (~10-30 minutes)'
        )
        parser.add_argument(
            '--ci',
            action='store_true',
            help='CI mode: non-interactive, structured output, strict exit codes'
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results as JSON (implies --ci)'
        )
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Attempt to auto-fix common issues'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show commands as they execute'
        )

    def run(self, args: Namespace) -> int:
        console = self.console
        project_root = Path.cwd()
        start_time = time.time()

        # Setup log file for debugging
        log_dir = project_root / ".tito" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"preflight_{timestamp}.log"
        self.log_lines: List[str] = []
        self._log(f"TinyTorch Preflight - {timestamp}")
        self._log(f"Project root: {project_root}")
        self._log("-" * 60)

        # Determine check level
        if args.release:
            level = "release"
            level_emoji = "🚀"
            level_desc = "Release Validation"
        elif args.full:
            level = "full"
            level_emoji = "🔍"
            level_desc = "Full Validation"
        elif args.quick:
            level = "quick"
            level_emoji = "⚡"
            level_desc = "Quick Checks"
        else:
            level = "standard"
            level_emoji = "✈️"
            level_desc = "Standard Preflight"

        is_ci = args.ci or args.json
        verbose = getattr(args, 'verbose', False)

        # Show header (unless JSON output)
        if not args.json:
            console.print(Panel(
                f"[bold cyan]{level_emoji} {level_desc}[/bold cyan]\n\n"
                f"Running verification checks before {'CI/CD' if is_ci else 'your next step'}...\n"
                f"[dim]Level: {level} | CI Mode: {is_ci} | Verbose: {verbose}[/dim]",
                title="TinyTorch Preflight",
                border_style="bright_cyan"
            ))
            console.print()

        # Run checks based on level
        categories = []

        # Level 1: Quick checks (always run)
        categories.append(self._check_structure(project_root, verbose))
        categories.append(self._check_cli(project_root, verbose))
        categories.append(self._check_imports(project_root, verbose))

        # Level 2: Standard checks
        if level in ["standard", "full", "release"]:
            categories.append(self._check_git_state(project_root, verbose))

        # Level 3: Full checks
        if level in ["full", "release"]:
            categories.append(self._check_module_tests(project_root, quick=(level != "release"), verbose=verbose))

        # Level 4: Release checks
        if level == "release":
            categories.append(self._check_milestones(project_root, verbose))
            categories.append(self._check_e2e(project_root, verbose))
            categories.append(self._check_docs(project_root, verbose))

        # Calculate totals
        total_passed = sum(c.passed for c in categories)
        total_failed = sum(c.failed for c in categories)
        total_warned = sum(c.warned for c in categories)
        total_checks = sum(len(c.checks) for c in categories)
        all_passed = total_failed == 0

        duration = time.time() - start_time

        # Save log file
        self._log("-" * 60)
        self._log(f"Completed in {duration:.2f}s")
        self._log(f"Result: {'PASS' if all_passed else 'FAIL'}")
        self._save_log()

        # Output results
        if args.json:
            self._output_json(categories, all_passed, duration)
        else:
            self._output_rich(categories, all_passed, duration, total_passed, total_failed, total_warned, total_checks, level, is_ci, verbose)

            # Show log location on failure
            if not all_passed:
                console.print(f"\n[dim]📋 Debug log: {self.log_file}[/dim]")

        return 0 if all_passed else 1

    def _log(self, message: str) -> None:
        """Add a line to the log."""
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _save_log(self) -> None:
        """Save the log file."""
        try:
            self.log_file.write_text("\n".join(self.log_lines))
        except Exception:
            pass  # Don't fail if we can't write log

    def _run_command(self, cmd: List[str], cwd: Path, timeout: int = 60, verbose: bool = False) -> Tuple[int, str, str]:
        """Run a command and return (exit_code, stdout, stderr)."""
        cmd_str = " ".join(cmd)
        self._log(f"Running: {cmd_str}")

        if verbose:
            self.console.print(f"[dim]  $ {cmd_str}[/dim]")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n')[:20]:  # First 20 lines
                    self._log(f"  stdout: {line}")
            if result.stderr.strip():
                for line in result.stderr.strip().split('\n')[:10]:  # First 10 lines
                    self._log(f"  stderr: {line}")
            self._log(f"  exit code: {result.returncode}")

            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self._log(f"  TIMEOUT after {timeout}s")
            return -1, "", "Command timed out"
        except Exception as e:
            self._log(f"  ERROR: {str(e)}")
            return -1, "", str(e)

    def _check_structure(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check project structure and required files."""
        category = CheckCategory(name="Project Structure", emoji="📁")

        # Required directories
        required_dirs = [
            ("modules/", "Module notebooks directory"),
            ("src/", "Source files directory"),
            ("milestones/", "Milestone scripts"),
            ("tests/", "Test directory"),
            ("tito/", "CLI directory"),
        ]

        # Optional directories (generated, not in git)
        optional_dirs = [
            ("tinytorch/", "Package directory (run 'tito export' to generate)"),
        ]

        for dir_path, desc in required_dirs:
            start = time.time()
            path = project_root / dir_path
            if path.exists() and path.is_dir():
                category.checks.append(CheckResult(
                    name=f"{dir_path} exists",
                    status=CheckStatus.PASS,
                    duration_ms=int((time.time() - start) * 1000)
                ))
            else:
                category.checks.append(CheckResult(
                    name=f"{dir_path} exists",
                    status=CheckStatus.FAIL,
                    message=f"Missing: {desc}",
                    duration_ms=int((time.time() - start) * 1000)
                ))

        # Optional directories (generated, warn if missing)
        for dir_path, desc in optional_dirs:
            start = time.time()
            path = project_root / dir_path
            if path.exists() and path.is_dir():
                category.checks.append(CheckResult(
                    name=f"{dir_path} exists",
                    status=CheckStatus.PASS,
                    duration_ms=int((time.time() - start) * 1000)
                ))
            else:
                category.checks.append(CheckResult(
                    name=f"{dir_path} exists",
                    status=CheckStatus.WARN,
                    message=desc,
                    duration_ms=int((time.time() - start) * 1000)
                ))

        # Required files
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            "README.md",
        ]

        for file_path in required_files:
            start = time.time()
            path = project_root / file_path
            if path.exists():
                category.checks.append(CheckResult(
                    name=f"{file_path} exists",
                    status=CheckStatus.PASS,
                    duration_ms=int((time.time() - start) * 1000)
                ))
            else:
                category.checks.append(CheckResult(
                    name=f"{file_path} exists",
                    status=CheckStatus.FAIL,
                    message=f"Missing required file",
                    duration_ms=int((time.time() - start) * 1000)
                ))

        # Check module count
        start = time.time()
        modules_dir = project_root / "modules"
        if modules_dir.exists():
            module_count = len([d for d in modules_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
            if module_count >= 15:
                category.checks.append(CheckResult(
                    name=f"Module count ({module_count})",
                    status=CheckStatus.PASS,
                    duration_ms=int((time.time() - start) * 1000)
                ))
            else:
                category.checks.append(CheckResult(
                    name=f"Module count ({module_count})",
                    status=CheckStatus.WARN,
                    message=f"Expected 20+ modules, found {module_count}",
                    duration_ms=int((time.time() - start) * 1000)
                ))

        return category

    def _check_cli(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check CLI commands work."""
        category = CheckCategory(name="CLI Commands", emoji="🖥️")

        if verbose:
            self.console.print(f"\n[bold]🖥️ CLI Commands[/bold]")

        cli_checks = [
            (["--version"], "tito --version"),
            (["--help"], "tito --help"),
            (["module", "status"], "tito module status"),
            (["system", "info"], "tito system info"),
            (["milestone", "list", "--simple"], "tito milestone list"),
        ]

        # Use bin/tito wrapper (no pip install required)
        tito_bin = project_root / "bin" / "tito"

        for args, name in cli_checks:
            start = time.time()
            cmd = [sys.executable, str(tito_bin)] + args
            cmd_str = f"./bin/tito {' '.join(args)}"

            code, stdout, stderr = self._run_command(cmd, project_root, timeout=30, verbose=verbose)

            duration = int((time.time() - start) * 1000)

            if code == 0:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.PASS,
                    duration_ms=duration,
                    command=cmd_str,
                    stdout=stdout[:500],  # Capture first 500 chars
                    stderr=stderr[:500]
                ))
                if verbose:
                    self.console.print(f"    [green]✓[/green] {name} [dim]({duration}ms)[/dim]")
            else:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    message=stderr[:100] if stderr else "Command failed",
                    duration_ms=duration,
                    command=cmd_str,
                    stdout=stdout[:1000],
                    stderr=stderr[:1000]
                ))
                if verbose:
                    self.console.print(f"    [red]✗[/red] {name}")
                    if stderr:
                        self.console.print(f"      [dim red]{stderr[:200]}[/dim red]")

        return category

    def _check_imports(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check that key imports work."""
        category = CheckCategory(name="Package Imports", emoji="📦")

        if verbose:
            self.console.print(f"\n[bold]📦 Package Imports[/bold]")

        imports = [
            ("import tinytorch", "tinytorch package"),
            ("from tinytorch import Tensor", "Tensor class"),
            ("from tito.main import TinyTorchCLI", "CLI class"),
        ]

        for import_stmt, name in imports:
            start = time.time()
            cmd = [sys.executable, "-c", import_stmt]
            code, stdout, stderr = self._run_command(cmd, project_root, timeout=10, verbose=verbose)

            duration = int((time.time() - start) * 1000)

            if code == 0:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.PASS,
                    duration_ms=duration,
                    command=import_stmt,
                    stderr=stderr
                ))
                if verbose:
                    self.console.print(f"    [green]✓[/green] {name}")
            else:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.WARN,
                    message="Import failed (may need export)",
                    duration_ms=duration,
                    command=import_stmt,
                    stderr=stderr[:500]
                ))
                if verbose:
                    self.console.print(f"    [yellow]⚠[/yellow] {name} - {stderr[:100] if stderr else 'failed'}")

        return category

    def _check_git_state(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check git repository state."""
        category = CheckCategory(name="Git State", emoji="🔀")

        # Check if git repo
        start = time.time()
        code, stdout, stderr = self._run_command(["git", "status", "--porcelain"], project_root)
        duration = int((time.time() - start) * 1000)

        if code != 0:
            category.checks.append(CheckResult(
                name="Git repository",
                status=CheckStatus.WARN,
                message="Not a git repository",
                duration_ms=duration
            ))
            return category

        category.checks.append(CheckResult(
            name="Git repository",
            status=CheckStatus.PASS,
            duration_ms=duration
        ))

        # Check for uncommitted changes
        start = time.time()
        if stdout.strip():
            lines = stdout.strip().split('\n')
            category.checks.append(CheckResult(
                name="Clean working tree",
                status=CheckStatus.WARN,
                message=f"{len(lines)} uncommitted changes",
                duration_ms=int((time.time() - start) * 1000)
            ))
        else:
            category.checks.append(CheckResult(
                name="Clean working tree",
                status=CheckStatus.PASS,
                duration_ms=int((time.time() - start) * 1000)
            ))

        # Check current branch
        start = time.time()
        code, stdout, stderr = self._run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], project_root)
        branch = stdout.strip() if code == 0 else "unknown"
        category.checks.append(CheckResult(
            name=f"Branch: {branch}",
            status=CheckStatus.PASS,
            duration_ms=int((time.time() - start) * 1000)
        ))

        return category

    def _check_module_tests(self, project_root: Path, quick: bool = True, verbose: bool = False) -> CheckCategory:
        """Run module tests."""
        category = CheckCategory(name="Module Tests", emoji="🧪")

        if verbose:
            self.console.print(f"\n[bold]🧪 Module Tests[/bold]")

        # Determine which tests to run
        if quick:
            test_targets = [
                ("tests/01_tensor/", "Module 01 tests"),
                ("tests/02_activations/", "Module 02 tests"),
            ]
        else:
            test_targets = [
                ("tests/", "All tests"),
            ]

        for test_path, name in test_targets:
            start = time.time()
            full_path = project_root / test_path

            if not full_path.exists():
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.SKIP,
                    message="Test directory not found",
                    duration_ms=0
                ))
                continue

            cmd = [sys.executable, "-m", "pytest", str(full_path), "-v", "--tb=short", "-q"]
            cmd_str = f"pytest {test_path} -v --tb=short -q"
            timeout = 300 if not quick else 60

            code, stdout, stderr = self._run_command(cmd, project_root, timeout=timeout, verbose=verbose)
            duration = int((time.time() - start) * 1000)

            if code == 0:
                passed_count = "all"
                for line in stdout.split('\n'):
                    if 'passed' in line:
                        passed_count = line.strip()
                        break

                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.PASS,
                    message=passed_count,
                    duration_ms=duration,
                    command=cmd_str,
                    stdout=stdout[-2000:],  # Last 2000 chars
                    stderr=stderr
                ))
                if verbose:
                    self.console.print(f"    [green]✓[/green] {name}: {passed_count}")
            else:
                failed_info = "Tests failed"
                for line in stdout.split('\n'):
                    if 'failed' in line.lower() or 'error' in line.lower():
                        failed_info = line.strip()[:80]
                        break

                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    message=failed_info,
                    duration_ms=duration,
                    command=cmd_str,
                    stdout=stdout[-3000:],  # More output on failure
                    stderr=stderr
                ))
                if verbose:
                    self.console.print(f"    [red]✗[/red] {name}: {failed_info}")
                    # Show last few lines of output
                    last_lines = stdout.strip().split('\n')[-5:]
                    for line in last_lines:
                        self.console.print(f"      [dim]{line}[/dim]")

        return category

    def _check_milestones(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check milestone scripts exist and are runnable."""
        category = CheckCategory(name="Milestones", emoji="🏆")

        if verbose:
            self.console.print(f"\n[bold]🏆 Milestones[/bold]")

        milestones_dir = project_root / "milestones"
        if not milestones_dir.exists():
            category.checks.append(CheckResult(
                name="Milestones directory",
                status=CheckStatus.FAIL,
                message="milestones/ not found"
            ))
            return category

        milestone_scripts = [
            ("01_1957_perceptron/02_rosenblatt_trained.py", "Perceptron script"),
            ("02_1969_xor/02_xor_solved.py", "XOR script"),
            ("03_1986_mlp/01_rumelhart_tinydigits.py", "MLP script"),
        ]

        for script_path, name in milestone_scripts:
            start = time.time()
            full_path = milestones_dir / script_path

            if full_path.exists():
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.PASS,
                    duration_ms=int((time.time() - start) * 1000)
                ))
                if verbose:
                    self.console.print(f"    [green]✓[/green] {name}")
            else:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.WARN,
                    message="Script not found",
                    duration_ms=int((time.time() - start) * 1000)
                ))
                if verbose:
                    self.console.print(f"    [yellow]⚠[/yellow] {name} - not found")

        return category

    def _check_e2e(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Run E2E tests."""
        category = CheckCategory(name="E2E Tests", emoji="🔄")

        if verbose:
            self.console.print(f"\n[bold]🔄 E2E Tests[/bold]")

        e2e_dir = project_root / "tests" / "e2e"
        if not e2e_dir.exists():
            category.checks.append(CheckResult(
                name="E2E test directory",
                status=CheckStatus.WARN,
                message="tests/e2e/ not found"
            ))
            return category

        start = time.time()
        cmd = [sys.executable, "-m", "pytest", str(e2e_dir), "-v", "-k", "quick", "--tb=short"]
        cmd_str = "pytest tests/e2e/ -v -k quick --tb=short"

        code, stdout, stderr = self._run_command(cmd, project_root, timeout=120, verbose=verbose)
        duration = int((time.time() - start) * 1000)

        if code == 0:
            category.checks.append(CheckResult(
                name="E2E quick tests",
                status=CheckStatus.PASS,
                duration_ms=duration,
                command=cmd_str,
                stdout=stdout[-2000:]
            ))
            if verbose:
                self.console.print(f"    [green]✓[/green] E2E quick tests passed")
        else:
            category.checks.append(CheckResult(
                name="E2E quick tests",
                status=CheckStatus.FAIL,
                message="E2E tests failed",
                duration_ms=duration,
                command=cmd_str,
                stdout=stdout[-3000:],
                stderr=stderr
            ))
            if verbose:
                self.console.print(f"    [red]✗[/red] E2E quick tests failed")
                last_lines = stdout.strip().split('\n')[-5:]
                for line in last_lines:
                    self.console.print(f"      [dim]{line}[/dim]")

        return category

    def _check_docs(self, project_root: Path, verbose: bool = False) -> CheckCategory:
        """Check documentation exists."""
        category = CheckCategory(name="Documentation", emoji="📚")

        doc_files = [
            ("README.md", "Main README"),
            ("docs/getting-started.md", "Getting Started"),
            ("CONTRIBUTING.md", "Contributing Guide"),
        ]

        for file_path, name in doc_files:
            start = time.time()
            full_path = project_root / file_path

            if full_path.exists():
                # Check it's not empty
                size = full_path.stat().st_size
                if size > 100:
                    category.checks.append(CheckResult(
                        name=name,
                        status=CheckStatus.PASS,
                        duration_ms=int((time.time() - start) * 1000)
                    ))
                else:
                    category.checks.append(CheckResult(
                        name=name,
                        status=CheckStatus.WARN,
                        message="File seems empty",
                        duration_ms=int((time.time() - start) * 1000)
                    ))
            else:
                category.checks.append(CheckResult(
                    name=name,
                    status=CheckStatus.WARN,
                    message="File not found",
                    duration_ms=int((time.time() - start) * 1000)
                ))

        return category

    def _output_json(self, categories: List[CheckCategory], all_passed: bool, duration: float):
        """Output results as JSON for CI/CD."""
        output = {
            "success": all_passed,
            "duration_seconds": round(duration, 2),
            "categories": []
        }

        for category in categories:
            cat_data = {
                "name": category.name,
                "passed": category.passed,
                "failed": category.failed,
                "warned": category.warned,
                "checks": []
            }
            for check in category.checks:
                cat_data["checks"].append({
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms
                })
            output["categories"].append(cat_data)

        print(json.dumps(output, indent=2))

    def _output_rich(self, categories: List[CheckCategory], all_passed: bool, duration: float,
                     total_passed: int, total_failed: int, total_warned: int, total_checks: int,
                     level: str, is_ci: bool, verbose: bool = False):
        """Output results with rich formatting."""
        console = self.console

        for category in categories:
            # Create table for category
            table = Table(
                show_header=False,
                box=None,
                padding=(0, 1),
                expand=True
            )
            table.add_column("Status", width=3)
            table.add_column("Check", style="bold")
            table.add_column("Message", style="dim")
            table.add_column("Time", style="dim", justify="right", width=8)

            for check in category.checks:
                if check.status == CheckStatus.PASS:
                    status_icon = "[green]✓[/green]"
                elif check.status == CheckStatus.FAIL:
                    status_icon = "[red]✗[/red]"
                elif check.status == CheckStatus.WARN:
                    status_icon = "[yellow]⚠[/yellow]"
                else:
                    status_icon = "[dim]○[/dim]"

                time_str = f"{check.duration_ms}ms" if check.duration_ms > 0 else ""
                table.add_row(status_icon, check.name, check.message, time_str)

            # Category header
            status_summary = f"[green]{category.passed}✓[/green]"
            if category.failed > 0:
                status_summary += f" [red]{category.failed}✗[/red]"
            if category.warned > 0:
                status_summary += f" [yellow]{category.warned}⚠[/yellow]"

            console.print(f"\n[bold]{category.emoji} {category.name}[/bold] {status_summary}")
            console.print(table)

            # Show failure details (always show for failures, not just verbose)
            for check in category.checks:
                if check.status == CheckStatus.FAIL and (check.stdout or check.stderr):
                    console.print(f"\n  [bold red]Failed: {check.name}[/bold red]")
                    if check.command:
                        console.print(f"  [dim]Command: {check.command}[/dim]")
                    if check.stderr:
                        console.print(f"  [dim red]Error:[/dim red]")
                        for line in check.stderr.strip().split('\n')[:5]:
                            console.print(f"    [dim]{line}[/dim]")
                    if check.stdout and not check.stderr:
                        console.print(f"  [dim]Output (last lines):[/dim]")
                        for line in check.stdout.strip().split('\n')[-5:]:
                            console.print(f"    [dim]{line}[/dim]")

        # Summary
        console.print()
        if all_passed:
            console.print(Panel(
                f"[bold green]✅ All preflight checks passed![/bold green]\n\n"
                f"[green]{total_passed}[/green] passed  "
                f"[yellow]{total_warned}[/yellow] warnings  "
                f"[dim]{duration:.1f}s[/dim]\n\n"
                f"[dim]Ready for: commit, PR, or {level} deployment[/dim]",
                title="Preflight Complete",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold red]❌ Preflight checks failed[/bold red]\n\n"
                f"[green]{total_passed}[/green] passed  "
                f"[red]{total_failed}[/red] failed  "
                f"[yellow]{total_warned}[/yellow] warnings  "
                f"[dim]{duration:.1f}s[/dim]\n\n"
                f"[dim]Fix the issues above before proceeding[/dim]",
                title="Preflight Failed",
                border_style="red"
            ))

        # Show next steps
        if not is_ci:
            if all_passed:
                if level == "quick":
                    console.print("\n[dim]💡 For thorough validation: tito dev preflight --full[/dim]")
                elif level == "standard":
                    console.print("\n[dim]💡 For release validation: tito dev preflight --release[/dim]")
            else:
                console.print("\n[dim]💡 Fix issues and re-run: tito dev preflight[/dim]")
