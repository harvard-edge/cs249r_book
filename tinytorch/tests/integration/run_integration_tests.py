#!/usr/bin/env python3
"""
Integration Test Runner for TinyTorch Modules
Provides detailed reporting without complex pytest plugin dependencies
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    test_class: str
    passed: bool
    error_msg: str = ""
    duration: float = 0.0

class IntegrationTestRunner:
    """Run integration tests with detailed reporting."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.module_name = ""

    def run_module_tests(self, module_number: str) -> Dict:
        """Run all tests for a specific module."""
        test_file = f"test_module_{module_number}"
        self.module_name = module_number

        try:
            # Import the test module
            test_module = importlib.import_module(f"tests.integration.{test_file}")

            # Find all test classes
            test_classes = [
                getattr(test_module, name)
                for name in dir(test_module)
                if name.startswith("Test") and isinstance(getattr(test_module, name), type)
            ]

            # Run tests from each class
            for test_class in test_classes:
                self._run_test_class(test_class)

            # Generate report
            return self._generate_report()

        except ImportError as e:
            return {
                "module": module_number,
                "status": "IMPORT_ERROR",
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "total": 0
            }

    def _run_test_class(self, test_class):
        """Run all test methods in a test class."""
        # Create instance
        try:
            instance = test_class()
        except Exception as e:
            self.results.append(TestResult(
                test_name="__init__",
                test_class=test_class.__name__,
                passed=False,
                error_msg=f"Failed to instantiate: {e}"
            ))
            return

        # Find test methods
        test_methods = [
            method for method in dir(instance)
            if method.startswith("test_") and callable(getattr(instance, method))
        ]

        # Run each test
        for method_name in test_methods:
            self._run_single_test(instance, test_class.__name__, method_name)

    def _run_single_test(self, instance, class_name: str, method_name: str):
        """Run a single test method."""
        import time
        start = time.time()

        try:
            method = getattr(instance, method_name)
            method()

            self.results.append(TestResult(
                test_name=method_name,
                test_class=class_name,
                passed=True,
                duration=time.time() - start
            ))

        except AssertionError as e:
            self.results.append(TestResult(
                test_name=method_name,
                test_class=class_name,
                passed=False,
                error_msg=f"Assertion failed: {e}",
                duration=time.time() - start
            ))

        except Exception as e:
            self.results.append(TestResult(
                test_name=method_name,
                test_class=class_name,
                passed=False,
                error_msg=f"Unexpected error: {e}\n{traceback.format_exc()}",
                duration=time.time() - start
            ))

    def _generate_report(self) -> Dict:
        """Generate detailed test report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        return {
            "module": self.module_name,
            "status": "PASSED" if failed == 0 else "FAILED",
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "duration": sum(r.duration for r in self.results),
            "timestamp": datetime.now().isoformat(),
            "details": self.results
        }

    def print_report(self, report: Dict):
        """Print formatted test report."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()

        # Header
        status_emoji = "✅" if report["status"] == "PASSED" else "❌"
        header = f"{status_emoji} Module {report['module']} Integration Tests"

        console.print(Panel(header, style="bold blue"))

        # Summary stats
        console.print(f"\n📊 Test Summary:")
        console.print(f"  • Total: {report['total']} tests")
        console.print(f"  • ✅ Passed: {report['passed']}")
        console.print(f"  • ❌ Failed: {report['failed']}")
        console.print(f"  • ⏱️  Duration: {report['duration']:.2f}s")

        if report.get("details"):
            # Detailed results table
            table = Table(title="\n📋 Detailed Results", box=box.ROUNDED)
            table.add_column("Test Class", style="cyan")
            table.add_column("Test Method", style="yellow")
            table.add_column("Status", justify="center")
            table.add_column("Duration", justify="right")

            for result in report["details"]:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                status_style = "green" if result.passed else "red"

                table.add_row(
                    result.test_class,
                    result.test_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    f"{result.duration:.3f}s"
                )

            console.print(table)

            # Show errors if any
            failures = [r for r in report["details"] if not r.passed]
            if failures:
                console.print("\n❌ [red]Failed Tests:[/red]")
                for failure in failures:
                    console.print(f"\n  • {failure.test_class}.{failure.test_name}")
                    if failure.error_msg:
                        # Truncate long error messages
                        error_lines = failure.error_msg.split('\n')
                        if len(error_lines) > 3:
                            console.print(f"    {error_lines[0]}")
                            console.print(f"    ...")
                            console.print(f"    {error_lines[-1]}")
                        else:
                            for line in error_lines:
                                console.print(f"    {line}")

        # Final status
        if report["status"] == "PASSED":
            console.print("\n🎉 [green bold]All integration tests passed![/green bold]")
            console.print("✨ Module is ready for capability demonstration!")
        else:
            console.print("\n⚠️  [red]Some tests failed. Please fix the issues above.[/red]")

        return report["status"] == "PASSED"


def main():
    """Run integration tests for specified module."""
    import argparse

    parser = argparse.ArgumentParser(description="Run TinyTorch integration tests")
    parser.add_argument("module", help="Module number (e.g., 05_dataloader)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    runner = IntegrationTestRunner()
    report = runner.run_module_tests(args.module)

    if not args.quiet:
        runner.print_report(report)

    # Return appropriate exit code
    sys.exit(0 if report["status"] == "PASSED" else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Default: run module 05 tests as example
        runner = IntegrationTestRunner()
        report = runner.run_module_tests("05_dataloader")
        runner.print_report(report)
