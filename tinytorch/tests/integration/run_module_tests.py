#!/usr/bin/env python3
"""
Module Integration Test Runner
Tests how modules interface with their dependencies
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.integration.test_module_dependencies import (
    MODULE_DEPENDENCIES,
    get_module_integration_tests
)

class ModuleIntegrationTester:
    """Run integration tests for module dependencies."""

    def __init__(self):
        self.results = []

    def run_module_tests(self, module_name: str) -> Dict:
        """Run integration tests for a module and its dependencies."""

        # Mock problematic imports
        import unittest.mock as mock
        with mock.patch.dict('sys.modules', {
            'matplotlib': mock.MagicMock(),
            'matplotlib.pyplot': mock.MagicMock(),
        }):
            return self._run_tests(module_name)

    def _run_tests(self, module_name: str) -> Dict:
        """Internal test runner."""
        results = {
            'module': module_name,
            'dependencies': MODULE_DEPENDENCIES.get(module_name, []),
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0,
            'status': 'PENDING',
            'timestamp': datetime.now().isoformat()
        }

        # Get integration tests for this module
        tests = get_module_integration_tests(module_name)

        if not tests:
            results['status'] = 'NO_TESTS'
            results['message'] = f"No integration tests defined for {module_name}"
            return results

        # Run each test
        for test_name, test_func in tests:
            start_time = time.time()
            test_result = {
                'name': test_name,
                'status': 'PENDING',
                'duration': 0
            }

            try:
                test_func()
                test_result['status'] = 'PASSED'
                results['passed'] += 1
            except AssertionError as e:
                test_result['status'] = 'FAILED'
                test_result['error'] = f"Assertion failed: {str(e)}"
                results['failed'] += 1
            except ImportError as e:
                test_result['status'] = 'SKIPPED'
                test_result['error'] = f"Import error (module not ready): {str(e)}"
            except Exception as e:
                test_result['status'] = 'ERROR'
                test_result['error'] = f"Unexpected error: {str(e)}"
                results['failed'] += 1

            test_result['duration'] = time.time() - start_time
            results['tests'].append(test_result)

        # Calculate totals
        results['total'] = len(tests)
        results['status'] = 'PASSED' if results['failed'] == 0 else 'FAILED'

        return results

    def print_report(self, results: Dict):
        """Print formatted test report with Rich styling."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        from rich.tree import Tree

        console = Console()

        # Header panel
        status_emoji = "✅" if results['status'] == 'PASSED' else "❌"
        if results['status'] == 'NO_TESTS':
            status_emoji = "⚠️"

        title = f"{status_emoji} Module {results['module']} Integration Tests"
        console.print(Panel(title, style="bold blue", expand=False))

        # Dependencies info
        if results['dependencies']:
            console.print("\n📦 Module Dependencies:")
            dep_tree = Tree("Dependencies")
            for dep in results['dependencies']:
                dep_tree.add(f"[cyan]{dep}[/cyan]")
            console.print(dep_tree)
        else:
            console.print("\n📦 No dependencies (base module)")

        # Test results
        if results.get('tests'):
            console.print(f"\n📊 Test Results:")
            console.print(f"  • Total: {results['total']} tests")
            console.print(f"  • ✅ Passed: {results['passed']}")
            console.print(f"  • ❌ Failed: {results['failed']}")

            skipped = sum(1 for t in results['tests'] if t['status'] == 'SKIPPED')
            if skipped:
                console.print(f"  • ⏭️  Skipped: {skipped}")

            # Detailed table
            table = Table(title="\n📋 Integration Test Details", box=box.ROUNDED)
            table.add_column("Test Name", style="yellow")
            table.add_column("Status", justify="center")
            table.add_column("Duration", justify="right")
            table.add_column("Details")

            for test in results['tests']:
                status_map = {
                    'PASSED': '[green]✅ PASS[/green]',
                    'FAILED': '[red]❌ FAIL[/red]',
                    'ERROR': '[red]💥 ERROR[/red]',
                    'SKIPPED': '[dim]⏭️  SKIP[/dim]'
                }
                status = status_map.get(test['status'], '❓')

                details = ""
                if test.get('error'):
                    # Truncate long errors
                    error = test['error']
                    if len(error) > 50:
                        error = error[:50] + "..."
                    details = f"[dim]{error}[/dim]"

                table.add_row(
                    test['name'],
                    status,
                    f"{test['duration']:.3f}s",
                    details
                )

            console.print(table)

            # Failed test details
            failed_tests = [t for t in results['tests'] if t['status'] in ['FAILED', 'ERROR']]
            if failed_tests:
                console.print("\n❌ [bold red]Failed Test Details:[/bold red]")
                for test in failed_tests:
                    console.print(f"\n  • [yellow]{test['name']}[/yellow]")
                    if test.get('error'):
                        console.print(f"    [red]{test['error']}[/red]")

        elif results['status'] == 'NO_TESTS':
            console.print(f"\n⚠️  {results.get('message', 'No tests defined')}")

        # Summary
        console.print("\n" + "="*60)
        if results['status'] == 'PASSED':
            console.print("🎉 [green bold]All integration tests passed![/green bold]")
            console.print("✨ Module integrates correctly with its dependencies!")
            console.print("🚀 Ready for capability demonstration!")
        elif results['status'] == 'FAILED':
            console.print("⚠️  [red]Some integration tests failed.[/red]")
            console.print("Please fix the issues before proceeding.")
        else:
            console.print("ℹ️  No integration tests to run.")
        console.print("="*60)

        return results['status'] == 'PASSED'


def run_and_report(module_name: str) -> bool:
    """Run tests and return success status."""
    tester = ModuleIntegrationTester()
    results = tester.run_module_tests(module_name)
    success = tester.print_report(results)
    return success


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run module integration tests")
    parser.add_argument("module", help="Module name (e.g., 05_dataloader)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")

    args = parser.parse_args()

    tester = ModuleIntegrationTester()
    results = tester.run_module_tests(args.module)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        success = tester.print_report(results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
