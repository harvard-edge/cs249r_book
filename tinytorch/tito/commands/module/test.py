"""
Module Test Command for TinyTorch CLI.

Provides comprehensive module testing functionality:
- Run individual module tests with educational output
- Three-phase testing: Inline → Module → Integration
- Display detailed test results with WHAT/WHY context
- Track test failures and successes

This enables students to verify their implementations and understand
what each test is checking and why it matters.

TESTING PHILOSOPHY:
==================
When a student runs `tito module test 05`, we want them to understand:
1. Does my implementation work? (Inline tests)
2. Does it handle edge cases? (Module tests with --tinytorch)
3. Does it integrate correctly with previous modules? (Integration tests)

Each phase builds confidence and understanding.
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console, Group
from rich.rule import Rule

from ..base import BaseCommand
from ...core.modules import get_module_mapping, normalize_module_number


class ModuleTestCommand(BaseCommand):
    """Command to test module implementations with educational output."""

    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run module tests to verify implementation"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add test command arguments."""
        parser.add_argument(
            "module_number",
            nargs="?",
            default=None,
            help="Module number to test (01, 02, etc.)",
        )
        parser.add_argument(
            "--all", action="store_true", help="Test all modules sequentially"
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Show detailed test output",
        )
        parser.add_argument(
            "--stop-on-fail",
            action="store_true",
            help="Stop testing if a module fails (only with --all)",
        )
        parser.add_argument(
            "--summary",
            action="store_true",
            help="Show only summary without running tests",
        )
        parser.add_argument(
            "--unit-only",
            action="store_true",
            help="Run only inline unit tests (skip pytest and integration)",
        )
        parser.add_argument(
            "--no-integration",
            action="store_true",
            help="Skip integration tests",
        )

    # Module mapping and normalization now imported from core.modules

    def run_inline_tests(
        self, module_name: str, module_number: str, verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Phase 1: Run inline unit tests from the module source file.

        These are the quick sanity checks embedded in the module itself,
        triggered by the if __name__ == "__main__" block.
        """
        console = self.console
        src_dir = self.config.project_root / "src"
        module_file = src_dir / module_name / f"{module_name}.py"

        if not module_file.exists():
            return False, f"Module file not found: {module_file}"

        try:
            result = subprocess.run(
                [sys.executable, str(module_file)],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                timeout=300,
            )

            if verbose:
                if result.stdout:
                    console.print("[dim]" + result.stdout + "[/dim]")
                if result.stderr:
                    console.print("[yellow]" + result.stderr + "[/yellow]")

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Test timeout (>5 minutes)"
        except Exception as e:
            return False, f"Test execution failed: {str(e)}"

    def run_module_pytest(
        self, module_name: str, module_number: str, verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Phase 2: Run pytest on module-specific tests with educational output.

        These tests use the --tinytorch flag to provide WHAT/WHY context
        for each test, helping students understand what's being checked.
        """
        console = self.console
        tests_dir = self.config.project_root / "tests" / module_name

        if not tests_dir.exists():
            # No module-specific tests - that's OK
            return True, "No module-specific tests found"

        try:
            # Run pytest with --tinytorch for educational output
            # Use --no-cov to avoid root pyproject.toml coverage requirements
            cmd = [
                sys.executable, "-m", "pytest",
                str(tests_dir),
                "--tinytorch",
                "-v" if verbose else "-q",
                "--tb=short",
                "--no-cov",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                timeout=300,
            )

            # Always show pytest output for educational value
            if result.stdout:
                console.print(result.stdout)
            if result.stderr and verbose:
                console.print("[yellow]" + result.stderr + "[/yellow]")

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr or result.stdout

        except subprocess.TimeoutExpired:
            return False, "Pytest timeout (>5 minutes)"
        except Exception as e:
            return False, f"Pytest execution failed: {str(e)}"

    def run_integration_tests(
        self, module_number: str, verbose: bool = False
    ) -> Tuple[bool, str]:
        """
        Phase 3: Run integration tests for modules 01 through N.

        This verifies that the student's implementation works correctly
        with all the previous modules they've built.
        """
        console = self.console
        integration_dir = self.config.project_root / "tests" / "integration"

        if not integration_dir.exists():
            return True, "No integration tests directory"

        # Find integration tests relevant to this module and earlier
        module_num = int(module_number)

        # Key integration test files that should run progressively
        relevant_tests = []

        # Map module numbers to relevant integration tests
        # Each module inherits tests from earlier modules (progressive testing)
        integration_test_map = {
            # Foundation modules (01-08)
            1: ["test_basic_integration.py"],
            2: ["test_basic_integration.py"],
            3: ["test_layers_integration.py"],
            4: ["test_loss_gradients.py"],
            5: ["test_dataloader_integration.py"],  # DataLoader
            6: ["test_gradient_flow.py"],  # Autograd
            7: ["test_training_flow.py"],  # Optimizers
            8: ["test_training_flow.py"],  # Training

            # Architecture modules (09-13)
            9: ["test_cnn_integration.py"],
            10: [],  # Tokenization: self-contained, no integration deps
            11: [],  # Embeddings: tested in NLP pipeline (module 12)
            12: ["test_nlp_pipeline_flow.py"],  # Attention
            13: ["test_nlp_pipeline_flow.py"],  # Transformers

            # Performance modules (14-19) - build on all previous
            # These use the same integration tests to ensure optimizations
            # don't break existing functionality
            14: [],  # Profiling: observational, no integration changes
            15: [],  # Quantization: tested in module-specific tests
            16: [],  # Compression: tested in module-specific tests
            17: [],  # Acceleration: tested in module-specific tests
            18: [],  # Memoization: tested in module-specific tests
            19: [],  # Benchmarking: tested in module-specific tests

            # Capstone (20) - runs comprehensive validation
            20: ["test_training_flow.py", "test_nlp_pipeline_flow.py", "test_cnn_integration.py"],
        }

        # Collect all relevant tests up to and including this module
        for i in range(1, module_num + 1):
            if i in integration_test_map:
                for test_file in integration_test_map[i]:
                    test_path = integration_dir / test_file
                    if test_path.exists() and str(test_path) not in relevant_tests:
                        relevant_tests.append(str(test_path))

        if not relevant_tests:
            return True, "No relevant integration tests for this module"

        try:
            # Use --no-cov to avoid root pyproject.toml coverage requirements
            cmd = [
                sys.executable, "-m", "pytest",
                *relevant_tests,
                "--tinytorch",
                "-v" if verbose else "-q",
                "--tb=short",
                "--no-cov",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
                timeout=600,  # 10 minute timeout for integration tests
            )

            if result.stdout:
                console.print(result.stdout)
            if result.stderr and verbose:
                console.print("[yellow]" + result.stderr + "[/yellow]")

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr or result.stdout

        except subprocess.TimeoutExpired:
            return False, "Integration tests timeout (>10 minutes)"
        except Exception as e:
            return False, f"Integration tests failed: {str(e)}"

    def test_module(
        self, module_name: str, module_number: str, verbose: bool = False,
        unit_only: bool = False, no_integration: bool = False
    ) -> Tuple[bool, str]:
        """
        Run comprehensive tests for a single module in three phases:

        Phase 1 - Inline Tests: Quick sanity checks from the module itself
        Phase 2 - Module Tests: Detailed pytest with educational output
        Phase 3 - Integration Tests: Verify compatibility with earlier modules

        Returns:
            (success, output) tuple
        """
        console = self.console
        all_passed = True
        all_output = []

        # Header
        console.print()
        console.print(Panel(
            f"[bold cyan]Testing Module {module_number}: {module_name}[/bold cyan]\n\n"
            "[dim]Three-phase testing ensures your implementation is correct,[/dim]\n"
            "[dim]handles edge cases, and integrates with previous modules.[/dim]",
            border_style="cyan",
        ))
        console.print()

        # ─────────────────────────────────────────────────────────────
        # Phase 1: Inline Unit Tests
        # ─────────────────────────────────────────────────────────────
        console.print(Rule("[bold yellow]Phase 1: Inline Unit Tests[/bold yellow]", style="yellow"))
        console.print("[dim]Running quick sanity checks from the module source...[/dim]")
        console.print()

        success, output = self.run_inline_tests(module_name, module_number, verbose)
        all_output.append(output)

        if success:
            console.print("[green]✓ Phase 1 PASSED: Inline unit tests[/green]")
        else:
            console.print("[red]✗ Phase 1 FAILED: Inline unit tests[/red]")
            if not verbose:
                console.print(f"[dim]{output[:500]}...[/dim]" if len(output) > 500 else f"[dim]{output}[/dim]")
            all_passed = False

        console.print()

        # Stop here if unit-only mode
        if unit_only:
            return all_passed, "\n".join(all_output)

        # ─────────────────────────────────────────────────────────────
        # Phase 2: Module Pytest Tests
        # ─────────────────────────────────────────────────────────────
        console.print(Rule("[bold blue]Phase 2: Module Tests (with educational output)[/bold blue]", style="blue"))
        console.print("[dim]Running pytest with WHAT/WHY context for each test...[/dim]")
        console.print()

        success, output = self.run_module_pytest(module_name, module_number, verbose)
        all_output.append(output)

        if success:
            console.print("[green]✓ Phase 2 PASSED: Module tests[/green]")
        else:
            console.print("[red]✗ Phase 2 FAILED: Module tests[/red]")
            all_passed = False

        console.print()

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Integration Tests (optional)
        # ─────────────────────────────────────────────────────────────
        if not no_integration:
            console.print(Rule("[bold magenta]Phase 3: Integration Tests[/bold magenta]", style="magenta"))
            console.print(f"[dim]Verifying Module {module_number} works with modules 01-{module_number}...[/dim]")
            console.print()

            success, output = self.run_integration_tests(module_number, verbose)
            all_output.append(output)

            if success:
                console.print("[green]✓ Phase 3 PASSED: Integration tests[/green]")
            else:
                console.print("[red]✗ Phase 3 FAILED: Integration tests[/red]")
                all_passed = False

            console.print()

        return all_passed, "\n".join(all_output)

    def test_all_modules(
        self, verbose: bool = False, stop_on_fail: bool = False
    ) -> int:
        """Test all modules sequentially."""
        console = self.console
        module_mapping = get_module_mapping()

        console.print()
        console.print(
            Panel(
                f"[bold cyan]Running All Module Tests[/bold cyan]\n\n"
                f"[bold]Testing {len(module_mapping)} modules sequentially[/bold]\n"
                f"  • Verbose: {'Yes' if verbose else 'No'}\n"
                f"  • Stop on failure: {'Yes' if stop_on_fail else 'No'}\n\n"
                f"[dim]This will take several minutes...[/dim]",
                title="🧪 Test All Modules",
                border_style="cyan",
            )
        )
        console.print()

        passed = []
        failed = []
        errors = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Testing modules...", total=len(module_mapping))

            for module_num, module_name in sorted(module_mapping.items()):
                progress.update(task, description=f"[cyan]Testing Module {module_num}: {module_name}...")
                success, output = self.test_module(module_name, module_num, verbose)
                progress.advance(task)

                if success:
                    passed.append((module_num, module_name))
                else:
                    failed.append((module_num, module_name))
                    errors[module_num] = output

                    if stop_on_fail:
                        console.print()
                        console.print(
                            Panel(
                                f"[red]Testing stopped due to failure in Module {module_num}[/red]\n\n"
                                f"[dim]Use --verbose to see full error details[/dim]",
                                title="Stopped on Failure",
                                border_style="red",
                            )
                        )
                        break

                console.print()

        # Display summary
        console.print()
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print("[bold cyan]Test Summary[/bold cyan]")
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print()

        # Create results table
        table = Table(title="Module Test Results", show_header=True)
        table.add_column("Module", style="cyan")
        table.add_column("Name", style="dim")
        table.add_column("Status", justify="center")

        for module_num, module_name in sorted(module_mapping.items()):
            if (module_num, module_name) in passed:
                status = "[green]✓ PASS[/green]"
            elif (module_num, module_name) in failed:
                status = "[red]✗ FAIL[/red]"
            else:
                status = "[dim]⏭ SKIPPED[/dim]"

            table.add_row(f"Module {module_num}", module_name, status)

        console.print(table)
        console.print()

        # Summary stats
        total = len(module_mapping)
        pass_count = len(passed)
        fail_count = len(failed)
        skip_count = total - pass_count - fail_count

        if fail_count == 0:
            console.print(
                Panel(
                    f"[bold green]✅ ALL TESTS PASSED![/bold green]\n\n"
                    f"[green]Passed: {pass_count}/{total} modules[/green]\n\n"
                    f"[bold]All TinyTorch modules are working correctly![/bold]",
                    title="🎉 Success",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    f"[bold red]❌ SOME TESTS FAILED[/bold red]\n\n"
                    f"[green]Passed: {pass_count} modules[/green]\n"
                    f"[red]Failed: {fail_count} modules[/red]\n"
                    + (f"[dim]Skipped: {skip_count} modules[/dim]\n" if skip_count > 0 else "")
                    + f"\n[bold]Failed modules:[/bold]\n"
                    + "\n".join([f"  • Module {num}: {name}" for num, name in failed]),
                    title="⚠️  Test Failures",
                    border_style="red",
                )
            )

            # Show error details for failed modules
            if errors and not verbose:
                console.print()
                console.print("[yellow]Failure details (run with --verbose for full output):[/yellow]")
                console.print()
                for module_num in sorted(errors.keys()):
                    console.print(f"[red]Module {module_num}:[/red]")
                    console.print(f"[dim]{errors[module_num][:500]}...[/dim]")
                    console.print()

            return 1

    def run(self, args: Namespace) -> int:
        """Execute the test command."""
        console = self.console

        # Handle --all (test all modules)
        if getattr(args, "all", False):
            return self.test_all_modules(
                verbose=args.verbose, stop_on_fail=args.stop_on_fail
            )

        # Require module number for single module test
        if not args.module_number:
            console.print(
                Panel(
                    "[red]Error: Module number required[/red]\n\n"
                    "[dim]Examples:[/dim]\n"
                    "[dim]  tito module test 01        # Test module 01[/dim]\n"
                    "[dim]  tito module test 01 -v     # Test with verbose output[/dim]\n"
                    "[dim]  tito module test --all     # Test all modules[/dim]",
                    title="Module Number Required",
                    border_style="red",
                )
            )
            return 1

        # Normalize and validate module number
        module_mapping = get_module_mapping()
        normalized = normalize_module_number(args.module_number)

        if normalized not in module_mapping:
            console.print(f"[red]Invalid module number: {args.module_number}[/red]")
            console.print("Available modules: 01-20")
            return 1

        module_name = module_mapping[normalized]

        # Test single module with enhanced three-phase testing
        success, output = self.test_module(
            module_name,
            normalized,
            verbose=args.verbose,
            unit_only=getattr(args, "unit_only", False),
            no_integration=getattr(args, "no_integration", False),
        )

        if success:
            console.print(
                Panel(
                    f"[bold green]✅ Module {normalized} - All Tests Passed![/bold green]\n\n"
                    f"[green]Your {module_name} implementation is working correctly[/green]\n"
                    f"[green]and integrates well with previous modules.[/green]",
                    title=f"✓ {module_name}",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    f"[bold red]❌ Module {normalized} - Some Tests Failed[/bold red]\n\n"
                    f"[yellow]Review the test output above to understand what failed.[/yellow]\n"
                    f"[dim]Each test includes WHAT it's checking and WHY it matters.[/dim]\n\n"
                    f"[dim]Tips:[/dim]\n"
                    f"[dim]  • Use -v flag for more detailed output[/dim]\n"
                    f"[dim]  • Use --unit-only to test just inline tests[/dim]\n"
                    f"[dim]  • Use --no-integration to skip integration tests[/dim]",
                    title=f"✗ {module_name}",
                    border_style="red",
                )
            )
            return 1
