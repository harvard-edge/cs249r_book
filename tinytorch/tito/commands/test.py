"""
Enhanced Test command for TinyTorch CLI: runs both inline and external tests.
"""

import subprocess
import sys
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from .base import BaseCommand

class TestResult:
    """Container for test results."""
    def __init__(self, name: str, success: bool, output: str = "", error: str = ""):
        self.name = name
        self.success = success
        self.output = output
        self.error = error

class ModuleTestResult:
    """Container for module test results."""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.inline_tests: List[TestResult] = []
        self.external_tests: List[TestResult] = []
        self.compilation_success = True
        self.compilation_error = ""

    @property
    def all_tests_passed(self) -> bool:
        """Check if all tests passed."""
        if not self.compilation_success:
            return False

        all_inline_passed = all(test.success for test in self.inline_tests)
        all_external_passed = all(test.success for test in self.external_tests)

        return all_inline_passed and all_external_passed

    @property
    def total_tests(self) -> int:
        """Total number of tests."""
        return len(self.inline_tests) + len(self.external_tests)

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        passed_inline = sum(1 for test in self.inline_tests if test.success)
        passed_external = sum(1 for test in self.external_tests if test.success)
        return passed_inline + passed_external

class TestCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "test"

    @property
    def description(self) -> str:
        return "Run module tests (inline and external)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("module", nargs="?", help="Module to test (optional)")
        parser.add_argument("--all", action="store_true", help="Run all module tests")
        parser.add_argument("--inline-only", action="store_true", help="Run only inline tests")
        parser.add_argument("--external-only", action="store_true", help="Run only external tests")
        parser.add_argument("--detailed", action="store_true", help="Show detailed output for all tests")
        parser.add_argument("--summary", action="store_true", help="Show summary report only")
        parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite (modules + integration + examples)")
        parser.add_argument("--student", action="store_true", help="Student-friendly comprehensive testing (same as --comprehensive)")

    def validate_args(self, args: Namespace) -> None:
        """Validate test command arguments."""
        if args.inline_only and args.external_only:
            raise ValueError("Cannot use --inline-only and --external-only together")

    def run(self, args: Namespace) -> int:
        console = self.console

        if args.comprehensive or args.student:
            return self._run_comprehensive_tests(args)
        elif args.all:
            return self._run_all_tests(args)
        elif args.module:
            return self._run_single_module_test(args)
        else:
            return self._show_available_modules()

    def _show_sync_reminder(self) -> None:
        """Show reminder to sync/export modules before testing."""
        console = self.console

        reminder_text = Text()
        reminder_text.append("💡 ", style="bright_yellow")
        reminder_text.append("Before running tests, make sure to sync modules to package:\n", style="bright_yellow")
        reminder_text.append("   • ", style="bright_cyan")
        reminder_text.append("tito export", style="bright_cyan bold")
        reminder_text.append(" - Export all modules to tinytorch package\n", style="bright_cyan")
        reminder_text.append("   • ", style="bright_cyan")
        reminder_text.append("tito nbdev build", style="bright_cyan bold")
        reminder_text.append(" - Build package with latest changes", style="bright_cyan")

        console.print(Panel(reminder_text,
                          title="Sync Reminder",
                          border_style="yellow",
                          padding=(0, 1)))
        console.print()  # Add spacing

    def _run_comprehensive_tests(self, args: Namespace) -> int:
        """Run comprehensive test suite (modules + integration + examples)."""
        console = self.console

        console.print(Panel.fit(
            "🎓 [bold blue]Student-Friendly Comprehensive Testing[/bold blue]\n"
            "[dim]This runs ALL tests to show your complete progress![/dim]",
            border_style="blue"
        ))

        try:
            # Run the comprehensive test runner script
            test_runner_path = Path("tests/run_all_modules.py")

            if not test_runner_path.exists():
                console.print("❌ [red]Comprehensive test runner not found![/red]")
                console.print(f"Expected: {test_runner_path}")
                return 1

            result = subprocess.run([
                sys.executable, str(test_runner_path)
            ], cwd=Path.cwd())

            return result.returncode

        except Exception as e:
            console.print(f"❌ [red]Error running comprehensive tests: {e}[/red]")
            return 1

    def _run_all_tests(self, args: Namespace) -> int:
        """Run tests for all modules."""
        console = self.console

        modules = self._discover_modules()
        if not modules:
            console.print(Panel("[yellow]⚠️  No modules found[/yellow]",
                              title="No Modules", border_style="yellow"))
            return 0

        console.print(Panel(f"🧪 Running tests for {len(modules)} modules",
                          title="Test Suite", border_style="bright_cyan"))

        # Show sync/export reminder before testing
        self._show_sync_reminder()

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            task = progress.add_task("Running tests...", total=len(modules))

            for i, module_name in enumerate(modules, 1):
                progress.update(task, description=f"Testing {module_name}... ({i}/{len(modules)})")

                # Show which module we're starting
                console.print(f"[cyan]🧪 Starting {module_name}...[/cyan]")

                result = self._test_module(module_name, args)
                results.append(result)

                # Show immediate feedback
                if result.all_tests_passed:
                    console.print(f"[green]✅ {module_name} - All tests passed ({result.passed_tests}/{result.total_tests})[/green]")
                else:
                    console.print(f"[red]❌ {module_name} - Tests failed ({result.passed_tests}/{result.total_tests})[/red]")

                progress.advance(task)

        # Generate report
        if args.summary:
            self._generate_summary_report(results)
        elif args.detailed:
            self._generate_detailed_report(results)
        else:
            self._generate_default_report(results)

        # Return success if all modules passed
        failed_modules = [r for r in results if not r.all_tests_passed]
        return 0 if not failed_modules else 1

    def _run_single_module_test(self, args: Namespace) -> int:
        """Run tests for a single module with detailed output."""
        console = self.console

        # Resolve module number to full name (e.g., "15" -> "15_quantization")
        module_name = self._resolve_module_name(args.module)

        console.print(Panel(f"🧪 Running tests for module: [bold cyan]{module_name}[/bold cyan]",
                          title="Single Module Test", border_style="bright_cyan"))

        # Show sync/export reminder before testing
        self._show_sync_reminder()

        result = self._test_module(module_name, args)

        # Always show detailed output for single module tests
        self._show_detailed_module_result(result)

        return 0 if result.all_tests_passed else 1

    def _test_module(self, module_name: str, args: Namespace) -> ModuleTestResult:
        """Test a single module comprehensively."""
        result = ModuleTestResult(module_name)
        console = self.console

        # Test compilation first
        dev_file = self._get_dev_file_path(module_name)
        if not dev_file.exists():
            result.compilation_success = False
            result.compilation_error = f"Module file not found: {dev_file}"
            return result

        # Test Python compilation
        console.print(f"[dim]  • Checking compilation...[/dim]")
        try:
            subprocess.run([sys.executable, "-m", "py_compile", str(dev_file)],
                          check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            result.compilation_success = False
            result.compilation_error = f"Compilation error: {e.stderr}"
            return result

        # Run inline tests if requested
        if not args.external_only:
            console.print(f"[dim]  • Running inline tests...[/dim]")
            inline_tests = self._run_inline_tests(dev_file)
            result.inline_tests = inline_tests

        # Run external tests if requested
        if not args.inline_only:
            console.print(f"[dim]  • Running external tests...[/dim]")
            external_tests = self._run_external_tests(module_name)
            result.external_tests = external_tests

        console.print(f"[dim]  • Completed {module_name} testing ({result.passed_tests}/{result.total_tests} tests passed)[/dim]")
        return result

    def _run_inline_tests(self, dev_file: Path) -> List[TestResult]:
        """Run inline tests using the module's standardized testing framework."""
        inline_tests = []

        # Set up environment to include current directory in PYTHONPATH
        import os
        import tempfile
        env = os.environ.copy()
        project_root = Path.cwd()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(project_root)

        # Handle .ipynb files by converting to Python first
        if dev_file.suffix == '.ipynb':
            try:
                # Create temporary Python file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                # Convert notebook to Python using jupytext
                convert_result = subprocess.run(
                    ['jupytext', '--to', 'py:percent', '--output', str(tmp_path), str(dev_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if convert_result.returncode != 0:
                    inline_tests.append(TestResult("notebook_conversion", False,
                                                  convert_result.stdout,
                                                  f"Failed to convert notebook: {convert_result.stderr}"))
                    return inline_tests

                # Use the converted Python file for execution
                exec_file = tmp_path
            except Exception as e:
                inline_tests.append(TestResult("notebook_conversion", False, "", str(e)))
                return inline_tests
        else:
            exec_file = dev_file

        try:
            result = subprocess.run(
                [sys.executable, str(exec_file)],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                env=env
            )

            output = result.stdout
            error = result.stderr

            # Check return code - this is the definitive test of success
            if result.returncode != 0:
                inline_tests.append(TestResult("script_execution", False, output, error))
            else:
                # Module executed successfully (return code 0)
                # This is the correct indicator of success, not output parsing
                inline_tests.append(TestResult("inline_tests", True, output))

        except subprocess.TimeoutExpired:
            inline_tests.append(TestResult("timeout", False, "", "Test execution timed out"))
        except Exception as e:
            inline_tests.append(TestResult("subprocess_error", False, "", str(e)))
        finally:
            # Clean up temporary file if we created one
            if dev_file.suffix == '.ipynb' and 'tmp_path' in locals():
                try:
                    tmp_path.unlink()
                except:
                    pass

        return inline_tests

    def _run_external_tests(self, module_name: str) -> List[TestResult]:
        """Run external pytest tests for a module."""
        external_tests = []

        # Extract short name from module directory name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name

        test_file = Path("tests") / f"test_{short_name}.py"

        if not test_file.exists():
            return external_tests

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=300
            )

            # Parse pytest output to extract individual test results
            test_results = self._parse_pytest_output(result.stdout, result.stderr)

            # If parsing fails, create a single result for the whole file
            if not test_results:
                success = result.returncode == 0
                external_tests.append(TestResult(
                    f"external_tests_{short_name}",
                    success,
                    result.stdout,
                    result.stderr
                ))
            else:
                external_tests.extend(test_results)

        except subprocess.TimeoutExpired:
            external_tests.append(TestResult("external_tests_timeout", False, "", "Tests timed out after 5 minutes"))
        except Exception as e:
            external_tests.append(TestResult("external_tests_error", False, "", str(e)))

        return external_tests



    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract individual test results."""
        test_results = []
        seen_tests = set()  # Avoid duplicate entries

        # Look for verbose pytest output lines like:
        # test_setup.py::TestPersonalInfo::test_name_validation PASSED
        # test_setup.py::TestPersonalInfo::test_email_validation FAILED
        lines = stdout.split('\n')

        for line in lines:
            line = line.strip()

            # Skip lines that are just status words without context
            if line in ['PASSED', 'FAILED', 'SKIPPED', '::']:
                continue

            # Look for test result lines
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                try:
                    # Split the line to extract components
                    parts = line.split()
                    if len(parts) >= 2:
                        test_path = parts[0]  # e.g., "test_setup.py::TestPersonalInfo::test_name_validation"
                        status = parts[1]     # e.g., "PASSED" or "FAILED"

                        # Skip if this is not a proper test path (must contain :: and not just be "FAILED")
                        if '::' not in test_path or test_path in ['PASSED', 'FAILED']:
                            continue

                        # Skip if we've already seen this test (avoid duplicates)
                        if test_path in seen_tests:
                            continue
                        seen_tests.add(test_path)

                        # Extract meaningful test name from path
                        path_parts = test_path.split('::')
                        if len(path_parts) >= 3:
                            # Format: file::Class::method -> "Class: method"
                            class_name = path_parts[1]
                            method_name = path_parts[2]

                            # Clean up names for better readability
                            clean_class = class_name.replace('Test', '').replace('test_', '')
                            clean_method = method_name.replace('test_', '').replace('_', ' ').title()

                            test_name = f"{clean_class}: {clean_method}"
                        elif len(path_parts) >= 2:
                            # Format: file::method -> "method"
                            method_name = path_parts[1]
                            test_name = method_name.replace('test_', '').replace('_', ' ').title()
                        else:
                            # Fallback to just the method name
                            test_name = path_parts[0]

                        success = status == 'PASSED'

                        # If failed, try to extract error details from subsequent lines or stderr
                        error_msg = ""
                        if not success:
                            # Look for error details in stderr
                            if stderr:
                                stderr_lines = stderr.split('\n')
                                for err_line in stderr_lines:
                                    err_line = err_line.strip()
                                    if any(keyword in err_line for keyword in ['FAILED', 'AssertionError', 'Error:', 'Exception']):
                                        # Extract meaningful part of error
                                        if '::' in err_line and test_path.split('::')[-1] in err_line:
                                            error_msg = err_line
                                            break
                                        elif 'AssertionError' in err_line or 'Error:' in err_line:
                                            error_msg = err_line
                                            break

                        test_results.append(TestResult(test_name, success, line, error_msg))
                except (IndexError, ValueError):
                    # If parsing fails, skip this line to avoid meaningless entries
                    continue

        # If no individual test results found, look for summary
        if not test_results:
            # Look for pytest summary lines
            for line in lines:
                if 'failed' in line.lower() and 'passed' in line.lower():
                    # Lines like "2 failed, 5 passed in 1.23s"
                    test_results.append(TestResult("pytest_summary", False, line, stderr))
                    break
                elif 'passed' in line.lower() and ('test' in line.lower() or 'ok' in line.lower()):
                    # Lines like "5 passed in 1.23s"
                    test_results.append(TestResult("pytest_summary", True, line))
                    break

        return test_results

    def _resolve_module_name(self, module_input: str) -> str:
        """Resolve module number or partial name to full module name.

        Examples:
            "15" -> "15_quantization"
            "01" -> "01_tensor"
            "tensor" -> "01_tensor"
            "15_quantization" -> "15_quantization" (unchanged)
        """
        # If already a full module name, return as-is
        if Path("modules").joinpath(module_input).exists():
            return module_input

        # Try to find by number prefix
        if module_input.isdigit():
            padded = f"{int(module_input):02d}"
            for module_dir in Path("modules").iterdir():
                if module_dir.is_dir() and module_dir.name.startswith(f"{padded}_"):
                    return module_dir.name

        # Try to find by name suffix
        for module_dir in Path("modules").iterdir():
            if module_dir.is_dir() and module_dir.name.endswith(f"_{module_input}"):
                return module_dir.name

        # Return as-is if no match found
        return module_input

    def _discover_modules(self) -> List[str]:
        """Discover available modules."""
        modules = []
        source_dir = Path("modules")

        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    # Check if dev file exists
                    dev_file = self._get_dev_file_path(module_dir.name)
                    if dev_file.exists():
                        modules.append(module_dir.name)

        return sorted(modules)

    def _get_dev_file_path(self, module_name: str) -> Path:
        """Get the path to a module's dev file."""
        # Extract short name from module directory name (e.g., "01_tensor" -> "tensor")
        short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name

        # Try .ipynb first (notebook format), then .py
        ipynb_path = Path("modules") / module_name / f"{short_name}.ipynb"
        if ipynb_path.exists():
            return ipynb_path

        # Fallback to .py file
        return Path("modules") / module_name / f"{short_name}.py"

    def _generate_summary_report(self, results: List[ModuleTestResult]) -> None:
        """Generate a summary report for all modules."""
        console = self.console

        # Summary table
        table = Table(title="Test Summary Report", show_header=True, header_style="bold cyan")
        table.add_column("Module", style="bold cyan", width=15)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Inline Tests", width=12, justify="center")
        table.add_column("External Tests", width=12, justify="center")
        table.add_column("Total", width=10, justify="center")

        total_modules = len(results)
        passed_modules = 0
        total_tests = 0
        total_passed = 0

        for result in results:
            status = "✅ PASS" if result.all_tests_passed else "❌ FAIL"
            if result.all_tests_passed:
                passed_modules += 1

            inline_status = f"{len([t for t in result.inline_tests if t.success])}/{len(result.inline_tests)}"
            external_status = f"{len([t for t in result.external_tests if t.success])}/{len(result.external_tests)}"

            total_tests += result.total_tests
            total_passed += result.passed_tests

            table.add_row(
                result.module_name,
                status,
                inline_status,
                external_status,
                f"{result.passed_tests}/{result.total_tests}"
            )

        console.print(table)

        # Overall summary
        console.print(f"\n📊 Overall Summary:")
        console.print(f"   Modules: {passed_modules}/{total_modules} passed")
        console.print(f"   Tests: {total_passed}/{total_tests} passed")
        console.print(f"   Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "   No tests found")

    def _generate_detailed_report(self, results: List[ModuleTestResult]) -> None:
        """Generate a detailed report for all modules."""
        console = self.console

        console.print(Panel("📋 Detailed Test Report", title="Test Results", border_style="bright_cyan"))

        for result in results:
            self._show_detailed_module_result(result)

    def _generate_default_report(self, results: List[ModuleTestResult]) -> None:
        """Generate the default report (between summary and detailed)."""
        console = self.console

        failed_modules = [r for r in results if not r.all_tests_passed]

        if failed_modules:
            console.print(Panel(f"[red]❌ {len(failed_modules)} modules failed[/red]",
                              title="Failed Modules", border_style="red"))

            for result in failed_modules:
                console.print(f"\n[red]❌ {result.module_name}[/red]:")

                if not result.compilation_success:
                    console.print(f"   [red]Compilation Error: {result.compilation_error}[/red]")

                failed_inline = [t for t in result.inline_tests if not t.success]
                failed_external = [t for t in result.external_tests if not t.success]

                if failed_inline:
                    console.print(f"   [red]Failed inline tests: {', '.join(t.name for t in failed_inline)}[/red]")

                if failed_external:
                    console.print(f"   [red]Failed external tests: {', '.join(t.name for t in failed_external)}[/red]")
        else:
            console.print(Panel("[green]✅ All modules passed![/green]",
                              title="Test Results", border_style="green"))

    def _show_detailed_module_result(self, result: ModuleTestResult) -> None:
        """Show detailed results for a single module."""
        console = self.console

        status_color = "green" if result.all_tests_passed else "red"
        status_icon = "✅" if result.all_tests_passed else "❌"

        console.print(f"\n[{status_color}]{status_icon} {result.module_name}[/{status_color}]")

        if not result.compilation_success:
            console.print(f"   [red]❌ Compilation failed: {result.compilation_error}[/red]")
            return

        # Show inline test results
        if result.inline_tests:
            console.print("   📝 Inline Tests:")
            for test in result.inline_tests:
                icon = "✅" if test.success else "❌"
                color = "green" if test.success else "red"
                console.print(f"      [{color}]{icon} {test.name}[/{color}]")

                if not test.success:
                    # Show meaningful error details
                    error_to_show = ""

                    if test.error and test.error.strip():
                        # Use the error field if available
                        error_to_show = test.error.strip()
                    elif test.output:
                        # Extract error from output
                        output_lines = test.output.split('\n')
                        for line in output_lines:
                            line = line.strip()
                            if any(keyword in line.lower() for keyword in ['error:', 'failed:', 'exception:', 'traceback']):
                                error_to_show = line
                                break

                        # If no specific error found, look for warning messages
                        if not error_to_show:
                            for line in output_lines:
                                line = line.strip()
                                if 'warning:' in line.lower() or 'deprecated' in line.lower():
                                    error_to_show = line
                                    break

                    # Show error details if found
                    if error_to_show:
                        # Don't truncate important error messages - show more context
                        if len(error_to_show) > 400:
                            error_to_show = error_to_show[:400] + "..."

                        # Distinguish between warnings and actual errors
                        if any(keyword in error_to_show.lower() for keyword in ['warning:', 'userwarning', 'deprecation']):
                            console.print(f"         [dim yellow]Warning: {error_to_show}[/dim yellow]")
                        else:
                            console.print(f"         [dim red]Error: {error_to_show}[/dim red]")
                    else:
                        console.print(f"         [dim red]Error: Test failed (see module output for details)[/dim red]")

        # Show external test results
        if result.external_tests:
            console.print("   🧪 External Tests:")
            failed_external_tests = []
            for test in result.external_tests:
                icon = "✅" if test.success else "❌"
                color = "green" if test.success else "red"
                console.print(f"      [{color}]{icon} {test.name}[/{color}]")

                if not test.success:
                    failed_external_tests.append(test)
                    if test.error and test.error.strip():
                        # Show error details for failed external tests
                        error_msg = test.error.strip()
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "..."
                        console.print(f"         [dim red]Error: {error_msg}[/dim red]")

            # Show debugging hint box for failed external tests
            if failed_external_tests:
                self._show_debug_hint_box(console, result.module_name, failed_external_tests)

        # Summary for this module
        console.print(f"   📊 Summary: {result.passed_tests}/{result.total_tests} tests passed")

    def _show_debug_hint_box(self, console: Console, module_name: str, failed_tests: List[TestResult]) -> None:
        """Show a debugging hint box with pytest commands for failed tests."""
        # Extract short name from module directory name
        if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))):
            short_name = module_name[3:]  # Remove "00_" prefix
        else:
            short_name = module_name

        test_file = f"tests/test_{short_name}.py"

        # Build hint text
        hint_text = Text()
        hint_text.append("🔍 Debug these failing tests:\n\n", style="bold yellow")

        # Add general debugging commands
        hint_text.append("• Run all tests with detailed output:\n", style="bright_yellow")
        hint_text.append(f"  pytest {test_file} -v --tb=long\n\n", style="bright_cyan")

        hint_text.append("• Run with print statements visible:\n", style="bright_yellow")
        hint_text.append(f"  pytest {test_file} -v -s\n\n", style="bright_cyan")

        # Add specific test commands if we have parsed test names
        if len(failed_tests) <= 5:  # Only show specific commands for a few tests
            hint_text.append("• Run specific failing tests:\n", style="bright_yellow")
            for test in failed_tests[:5]:  # Limit to 5 tests
                # Convert display name back to pytest format
                # Handle format like "TemporaryFailures: Demo Failure One" -> "TestTemporaryFailures::test_demo_failure_one"
                if ":" in test.name:
                    class_part, method_part = test.name.split(":", 1)
                    class_name = f"Test{class_part.strip().replace(' ', '')}"
                    method_name = f"test_{method_part.strip().replace(' ', '_').lower()}"
                    test_path = f"{class_name}::{method_name}"
                else:
                    # Fallback for other formats
                    test_path = test.name.replace(" ", "_").replace(":", "").lower()
                hint_text.append(f"  pytest {test_file}::{test_path} -v -s\n", style="bright_cyan")

        # Add extra debugging tips
        hint_text.append("\n💡 Pro tips:\n", style="bold yellow")
        hint_text.append("• Use --pdb to drop into debugger on failure\n", style="white")
        hint_text.append("• Use -k 'test_name' to run tests by name pattern\n", style="white")
        hint_text.append("• Use --tb=short for concise error messages\n", style="white")

        console.print(Panel(hint_text, title="🐛 Debug Failed Tests", border_style="yellow"))

    def _show_available_modules(self) -> int:
        """Show available modules when no arguments are provided."""
        console = self.console

        modules = self._discover_modules()

        if modules:
            console.print(Panel(f"[red]❌ Please specify a module to test[/red]\n\n"
                              f"Available modules: {', '.join(modules)}\n\n"
                              f"[dim]Examples:[/dim]\n"
                              f"[dim]  tito module test tensor       - Test specific module[/dim]\n"
                              f"[dim]  tito module test --all        - Test all modules[/dim]\n"
                              f"[dim]  tito module test --all --summary - Summary report[/dim]",
                              title="Module Required", border_style="red"))
        else:
            console.print(Panel("[red]❌ No modules found in modules directory[/red]",
                              title="Error", border_style="red"))

        return 1
