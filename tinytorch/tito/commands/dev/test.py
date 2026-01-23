"""
Unified Developer Test Command for TinyTorch.

Simple, explicit test types:
    tito dev test                 # Default: unit tests
    tito dev test --unit          # Unit tests only
    tito dev test --integration   # Integration tests
    tito dev test --e2e           # End-to-end tests
    tito dev test --all           # All test types
    tito dev test --release       # Full release validation (destructive)

Think like PyTorch: explicit, predictable, one way to do things.
"""

import subprocess
import sys
import time
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box

from ..base import BaseCommand


@dataclass
class TestResult:
    """Result of a test phase."""
    name: str
    passed: bool
    duration: float = 0.0
    message: str = ""
    test_count: int = 0


class DevTestCommand(BaseCommand):
    """Unified developer testing command."""

    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run tests: --unit, --integration, --e2e, --all, --release"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add test command arguments."""
        # Test type flags (can combine multiple)
        parser.add_argument(
            "--unit", "-u",
            action="store_true",
            help="Run unit tests (module-level)"
        )
        parser.add_argument(
            "--integration", "-i",
            action="store_true",
            help="Run integration tests"
        )
        parser.add_argument(
            "--e2e", "-e",
            action="store_true",
            help="Run end-to-end tests"
        )
        parser.add_argument(
            "--cli",
            action="store_true",
            help="Run CLI tests"
        )
        parser.add_argument(
            "--all", "-a",
            action="store_true",
            help="Run all test types"
        )
        parser.add_argument(
            "--release",
            action="store_true",
            help="Full release validation (destructive - resets all modules)"
        )
        parser.add_argument(
            "--milestone",
            action="store_true",
            help="Run milestone tests (validates milestone scripts execute)"
        )
        parser.add_argument(
            "--inline",
            action="store_true",
            help="Run inline tests from src/ (progressive: test + export each module)"
        )

        # Options
        parser.add_argument(
            "--module", "-m",
            type=str,
            metavar="N",
            help="Test specific module (e.g., -m 06)"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Show detailed output"
        )
        parser.add_argument(
            "--ci",
            action="store_true",
            help="CI mode: JSON output, strict exit codes"
        )
        parser.add_argument(
            "--no-build",
            action="store_true",
            help="Skip package build (assumes already exported)"
        )

    def run(self, args: Namespace) -> int:
        """Run the test suite."""
        console = self.console
        project_root = self.config.project_root
        start_time = time.time()

        # Determine what tests to run
        run_inline = args.inline or args.all
        run_unit = args.unit or args.all or (not any([args.unit, args.integration, args.e2e, args.cli, args.all, args.release, args.milestone, args.inline]))
        run_integration = args.integration or args.all
        run_e2e = args.e2e or args.all
        run_cli = args.cli or args.all
        run_release = args.release
        run_milestone = args.milestone or args.all

        # Build test type list for display
        test_types = []
        if run_inline:
            test_types.append("inline")
        if run_unit:
            test_types.append("unit")
        if run_integration:
            test_types.append("integration")
        if run_e2e:
            test_types.append("e2e")
        if run_cli:
            test_types.append("cli")
        if run_milestone:
            test_types.append("milestone")
        if run_release:
            test_types.append("release")

        # Header
        if not args.ci:
            console.print()
            test_desc = ", ".join(test_types) if test_types else "unit"
            module_desc = f" (module {args.module})" if args.module else ""
            console.print(Panel(
                f"[bold cyan]🧪 Running: {test_desc}{module_desc}[/bold cyan]\n\n"
                f"[bold]Test Types:[/bold]\n"
                f"  [bold]--inline[/bold]           Inline tests from src/ (progressive)\n"
                f"  [bold]--unit[/bold] (-u)        Pytest unit tests\n"
                f"  [bold]--integration[/bold] (-i) Cross-module integration tests\n"
                f"  [bold]--e2e[/bold] (-e)         End-to-end user journey tests\n"
                f"  [bold]--cli[/bold]              CLI command tests\n"
                f"  [bold]--milestone[/bold]        Milestone script tests\n"
                f"  [bold]--all[/bold] (-a)         All of the above\n"
                f"  [bold]--release[/bold]          Full validation (destructive)\n\n"
                f"[bold]Options:[/bold]\n"
                f"  [bold]-m N[/bold]               Test specific module\n"
                f"  [bold]--no-build[/bold]         Skip export (assume already built)\n"
                f"  [bold]--ci[/bold]               JSON output for automation",
                title="🔥 TinyTorch Developer Tests",
                border_style="cyan"
            ))
            console.print()

        results: List[TestResult] = []

        # =====================================================================
        # Step 1: Build Package (unless --no-build, release, or inline mode)
        # =====================================================================
        # Skip build for:
        # - --no-build: User explicitly skips
        # - --release: Will reset and rebuild each module
        # - --inline: Will test and export each module progressively
        if not args.no_build and not run_release and not run_inline:
            if not args.ci:
                console.print("[bold]Step 1: Build Package[/bold]")

            # For milestone tests, we need ALL modules exported
            # For other tests, a quick import check is sufficient
            if run_milestone:
                # Milestone tests require full package - always rebuild
                if not args.ci:
                    console.print("  [dim]Milestone tests require full package export...[/dim]")
                result = self._build_package(project_root, args.verbose)
                results.append(result)
                if not args.ci:
                    self._print_result(result)
                if not result.passed:
                    return self._finish(results, start_time, args)
            else:
                # Quick import check for other test types
                import_ok = self._check_imports(project_root)
                if import_ok:
                    if not args.ci:
                        console.print("  [green]✓[/green] Package already built")
                else:
                    result = self._build_package(project_root, args.verbose)
                    results.append(result)
                    if not args.ci:
                        self._print_result(result)
                    if not result.passed:
                        return self._finish(results, start_time, args)

            if not args.ci:
                console.print()

        # =====================================================================
        # Step 2: Run requested test types
        # =====================================================================

        # Inline tests run first (they build the package progressively)
        if run_inline:
            if not args.ci:
                console.print("[bold]Running: Inline Tests (progressive module build)[/bold]")
            result = self._run_inline_tests(project_root, args.module, args.verbose, args.ci)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()
            # If inline tests fail, stop here - package isn't fully built
            if not result.passed:
                return self._finish(results, start_time, args)

        if run_unit:
            if not args.ci:
                console.print("[bold]Running: Unit Tests[/bold]")
            result = self._run_unit_tests(project_root, args.module, args.verbose)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        if run_cli:
            if not args.ci:
                console.print("[bold]Running: CLI Tests[/bold]")
            result = self._run_cli_tests(project_root, args.verbose)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        if run_integration:
            if not args.ci:
                console.print("[bold]Running: Integration Tests[/bold]")
            result = self._run_integration_tests(project_root, args.verbose)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        if run_e2e:
            if not args.ci:
                console.print("[bold]Running: E2E Tests[/bold]")
            result = self._run_e2e_tests(project_root, args.verbose)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        if run_milestone:
            if not args.ci:
                console.print("[bold]Running: Milestone Tests[/bold]")
            result = self._run_milestone_tests(project_root, args.verbose)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        if run_release:
            if not args.ci:
                console.print("[bold]Running: Release Validation[/bold]")
                console.print("[yellow]⚠️  This will reset and rebuild ALL modules![/yellow]")
            result = self._run_release_validation(project_root, args)
            results.append(result)
            if not args.ci:
                self._print_result(result)
                console.print()

        return self._finish(results, start_time, args)

    def _print_result(self, result: TestResult) -> None:
        """Print a single test result."""
        if result.passed:
            count = f" ({result.test_count} tests)" if result.test_count else ""
            self.console.print(f"  [green]✓[/green] {result.name}{count} [dim]({result.duration:.1f}s)[/dim]")
        else:
            self.console.print(f"  [red]✗[/red] {result.name} [dim]({result.duration:.1f}s)[/dim]")
            if result.message:
                self.console.print(f"    [dim red]{result.message}[/dim red]")

    def _check_imports(self, project_root: Path) -> bool:
        """Quick check if package is already built."""
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from tinytorch import Tensor; assert Tensor is not None"],
                cwd=project_root,
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _build_package(self, project_root: Path, verbose: bool) -> TestResult:
        """Build package by exporting all modules."""
        start = time.time()

        try:
            cmd = [sys.executable, str(project_root / "bin" / "tito"), "dev", "export", "--all"]
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                return TestResult(
                    name="Build package",
                    passed=True,
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="Build package",
                    passed=False,
                    duration=time.time() - start,
                    message=result.stderr[:200] if result.stderr else "Export failed"
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                name="Build package",
                passed=False,
                duration=time.time() - start,
                message="Timed out after 5 minutes"
            )
        except Exception as e:
            return TestResult(
                name="Build package",
                passed=False,
                duration=time.time() - start,
                message=str(e)[:100]
            )

    def _run_pytest(self, project_root: Path, test_path: str, name: str,
                    verbose: bool, timeout: int = 300, extra_args: List[str] = None) -> TestResult:
        """Run pytest on a path and return result."""
        start = time.time()
        full_path = project_root / test_path

        if not full_path.exists():
            return TestResult(
                name=name,
                passed=True,
                duration=0,
                message="No tests found"
            )

        try:
            cmd = [
                sys.executable, "-m", "pytest",
                str(full_path),
                "-v" if verbose else "-q",
                "--tb=short",
                "--no-cov",
            ]
            if extra_args:
                cmd.extend(extra_args)

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Count tests from output
            test_count = 0
            summary = ""
            for line in result.stdout.split('\n'):
                if 'passed' in line:
                    summary = line.strip()
                    # Try to extract count
                    import re
                    match = re.search(r'(\d+) passed', line)
                    if match:
                        test_count = int(match.group(1))
                    break

            if result.returncode == 0:
                return TestResult(
                    name=name,
                    passed=True,
                    duration=time.time() - start,
                    test_count=test_count,
                    message=summary
                )
            else:
                # Extract failure info
                for line in result.stdout.split('\n'):
                    if 'failed' in line.lower() or 'error' in line.lower():
                        summary = line.strip()[:80]
                        break
                return TestResult(
                    name=name,
                    passed=False,
                    duration=time.time() - start,
                    message=summary or "Tests failed"
                )
        except subprocess.TimeoutExpired:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                message=f"Timed out after {timeout//60} minutes"
            )
        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration=time.time() - start,
                message=str(e)[:100]
            )

    def _run_inline_tests(self, project_root: Path, module: Optional[str],
                          verbose: bool, ci_mode: bool) -> TestResult:
        """Run inline tests from src/ files progressively.

        This simulates the student journey:
        1. For each module in order (01 → 20):
           a. Run inline tests from src/XX_module/XX_module.py
           b. If tests pass, export to tinytorch/core/
           c. If tests fail, stop and report
        """
        from ...core.modules import get_module_mapping

        start = time.time()
        console = self.console
        module_mapping = get_module_mapping()

        # Determine which modules to test
        if module:
            module_num = module.zfill(2)
            if module_num not in module_mapping:
                return TestResult(
                    name=f"Inline tests (module {module_num})",
                    passed=False,
                    duration=0,
                    message=f"Module {module_num} not found"
                )
            # Test up to and including the specified module
            target_int = int(module_num)
            module_nums = [m for m in sorted(module_mapping.keys(), key=lambda x: int(x))
                          if int(m) <= target_int]
        else:
            module_nums = sorted(module_mapping.keys(), key=lambda x: int(x))

        passed_modules = 0
        failed_module = None

        for module_num in module_nums:
            module_name = module_mapping[module_num]

            if not ci_mode:
                console.print(f"  [dim]Module {module_num} ({module_name})...[/dim]")

            # Run tito module complete (tests + export)
            try:
                result = subprocess.run(
                    [sys.executable, str(project_root / "bin" / "tito"),
                     "module", "complete", module_num],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=300  # 5 min per module
                )

                if result.returncode == 0:
                    passed_modules += 1
                    if not ci_mode:
                        console.print(f"    [green]✓[/green] Passed")
                else:
                    failed_module = f"{module_num}:{module_name}"
                    if not ci_mode:
                        console.print(f"    [red]✗[/red] Failed")
                        # Show error summary
                        for line in result.stdout.split('\n')[-10:]:
                            if line.strip():
                                console.print(f"      [dim red]{line}[/dim red]")
                    break

            except subprocess.TimeoutExpired:
                failed_module = f"{module_num}:timeout"
                break
            except Exception as e:
                failed_module = f"{module_num}:{str(e)[:30]}"
                break

        duration = time.time() - start

        if failed_module:
            return TestResult(
                name="Inline tests",
                passed=False,
                duration=duration,
                test_count=passed_modules,
                message=f"Failed at {failed_module}"
            )
        else:
            return TestResult(
                name="Inline tests",
                passed=True,
                duration=duration,
                test_count=passed_modules,
                message=f"{passed_modules}/{len(module_nums)} modules passed"
            )

    def _run_unit_tests(self, project_root: Path, module: Optional[str], verbose: bool) -> TestResult:
        """Run unit tests."""
        if module:
            module_num = module.zfill(2)
            test_dirs = list((project_root / "tests").glob(f"{module_num}_*"))
            if not test_dirs:
                return TestResult(
                    name=f"Unit tests (module {module_num})",
                    passed=True,
                    duration=0,
                    message="No tests found for this module"
                )
            test_path = str(test_dirs[0].relative_to(project_root))
            name = f"Unit tests (module {module_num})"
        else:
            test_path = "tests"
            name = "Unit tests"

        return self._run_pytest(
            project_root, test_path, name, verbose,
            extra_args=["--ignore=tests/e2e/", "--ignore=tests/integration/", "--ignore=tests/cli/"]
        )

    def _run_cli_tests(self, project_root: Path, verbose: bool) -> TestResult:
        """Run CLI tests."""
        return self._run_pytest(project_root, "tests/cli", "CLI tests", verbose, timeout=120)

    def _run_integration_tests(self, project_root: Path, verbose: bool) -> TestResult:
        """Run integration tests."""
        return self._run_pytest(project_root, "tests/integration", "Integration tests", verbose)

    def _run_e2e_tests(self, project_root: Path, verbose: bool) -> TestResult:
        """Run E2E tests."""
        return self._run_pytest(
            project_root, "tests/e2e", "E2E tests", verbose,
            timeout=600, extra_args=["-m", "quick"]
        )

    def _run_milestone_tests(self, project_root: Path, verbose: bool) -> TestResult:
        """Run milestone tests from tests/milestones/ directory.

        These are pytest-based tests that verify milestone scripts execute correctly.
        Requires the package to be fully exported with all modules completed.
        """
        return self._run_pytest(
            project_root, "tests/milestones", "Milestone tests", verbose,
            timeout=900  # 15 min for all milestones
        )

    def _run_release_validation(self, project_root: Path, args: Namespace) -> TestResult:
        """Run full release validation (destructive).

        This simulates the complete student journey:
        1. Reset (clear modules/ and tinytorch/core/)
        2. For each module: export from src, then run module complete
        3. Run all milestones
        """
        import shutil
        from ..milestone import MILESTONE_SCRIPTS
        from ...core.modules import get_module_mapping

        start = time.time()
        console = self.console

        # Get module list
        module_mapping = get_module_mapping()
        module_nums = sorted(module_mapping.keys(), key=lambda x: int(x))

        # =====================================================================
        # Step 1: Reset to clean state
        # =====================================================================
        if not args.ci:
            console.print("  [dim]Resetting to clean state...[/dim]")

        try:
            # Clear modules/ (remove all module subdirectories)
            modules_dir = project_root / "modules"
            if modules_dir.exists():
                for item in modules_dir.iterdir():
                    if item.is_dir() and item.name[0].isdigit():
                        shutil.rmtree(item)

            # Clear tinytorch/core/ (remove all .py except __init__.py)
            core_dir = project_root / "tinytorch" / "core"
            if core_dir.exists():
                for py_file in core_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        py_file.unlink()
        except Exception as e:
            return TestResult(
                name="Release validation",
                passed=False,
                duration=time.time() - start,
                message=f"Reset failed: {str(e)[:50]}"
            )

        # =====================================================================
        # Step 2: Validate all modules
        # =====================================================================
        failed_modules = []
        passed_modules = 0

        for module_num in module_nums:
            module_name = module_mapping[module_num]
            if not args.ci:
                console.print(f"  [dim]Module {module_num} ({module_name})...[/dim]")

            # Export from src
            try:
                result = subprocess.run(
                    [sys.executable, str(project_root / "bin" / "tito"), "dev", "export", module_num],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=120
                )
                if result.returncode != 0:
                    failed_modules.append(f"{module_num}:export")
                    continue
            except Exception:
                failed_modules.append(f"{module_num}:export")
                continue

            # Run module complete
            try:
                result = subprocess.run(
                    [sys.executable, str(project_root / "bin" / "tito"), "module", "complete", module_num],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=300
                )
                if result.returncode != 0:
                    failed_modules.append(f"{module_num}:complete")
                    continue
            except Exception:
                failed_modules.append(f"{module_num}:complete")
                continue

            passed_modules += 1

        # =====================================================================
        # Step 3: Run all milestones
        # =====================================================================
        failed_milestones = []
        passed_milestones = 0

        if not args.ci:
            console.print("  [dim]Running milestones...[/dim]")

        for milestone_id in sorted(MILESTONE_SCRIPTS.keys()):
            try:
                result = subprocess.run(
                    [sys.executable, str(project_root / "bin" / "tito"), "milestone", "run", milestone_id, "--skip-checks"],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=180
                )
                if result.returncode == 0:
                    passed_milestones += 1
                else:
                    failed_milestones.append(milestone_id)
            except Exception:
                failed_milestones.append(milestone_id)

        # Build result
        total_time = time.time() - start
        all_passed = len(failed_modules) == 0 and len(failed_milestones) == 0

        if all_passed:
            return TestResult(
                name="Release validation",
                passed=True,
                duration=total_time,
                test_count=passed_modules + passed_milestones,
                message=f"{passed_modules} modules, {passed_milestones} milestones"
            )
        else:
            failures = []
            if failed_modules:
                failures.append(f"modules: {', '.join(failed_modules[:3])}")
            if failed_milestones:
                failures.append(f"milestones: {', '.join(failed_milestones)}")
            return TestResult(
                name="Release validation",
                passed=False,
                duration=total_time,
                message="; ".join(failures)[:100]
            )

    def _finish(self, results: List[TestResult], start_time: float, args: Namespace) -> int:
        """Show final summary and return exit code."""
        console = self.console
        total_time = time.time() - start_time

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total_tests = sum(r.test_count for r in results)
        all_passed = failed == 0

        if args.ci:
            # JSON output for CI
            output = {
                "success": all_passed,
                "duration_seconds": round(total_time, 2),
                "passed": passed,
                "failed": failed,
                "total_tests": total_tests,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration": round(r.duration, 2),
                        "test_count": r.test_count,
                        "message": r.message
                    }
                    for r in results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            # Rich summary
            if all_passed:
                test_info = f"{total_tests} tests" if total_tests else f"{passed} phases"
                console.print(Panel(
                    f"[bold green]✅ ALL TESTS PASSED[/bold green]\n\n"
                    f"[green]{test_info}[/green] completed in [dim]{total_time:.1f}s[/dim]",
                    title="🎉 Success",
                    border_style="green"
                ))
            else:
                failed_names = [r.name for r in results if not r.passed]
                console.print(Panel(
                    f"[bold red]❌ TESTS FAILED[/bold red]\n\n"
                    f"[green]{passed}[/green] passed  [red]{failed}[/red] failed  [dim]{total_time:.1f}s[/dim]\n\n"
                    f"Failed: {', '.join(failed_names)}",
                    title="⚠️ Test Failures",
                    border_style="red"
                ))

        return 0 if all_passed else 1
