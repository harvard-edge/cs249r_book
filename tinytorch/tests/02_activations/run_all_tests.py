#!/usr/bin/env python3
"""
Run all tests for Module XX: [Module Name]
Template test runner - copy to each module's test directory
"""

import sys
from pathlib import Path
import importlib.util
import time
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_module_tests() -> Dict:
    """Run all tests for this module."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel

    console = Console()

    # Update module number and name
    MODULE_NUMBER = "03"
    MODULE_NAME = "Activations"

    # Header
    console.print(Panel(f"[bold blue]Module {MODULE_NUMBER}: {MODULE_NAME} - Test Suite[/bold blue]",
                       expand=False))

    # Find all test files in this module
    test_files = list(Path(__file__).parent.glob("test_*.py"))
    test_files = [f for f in test_files if f.name != Path(__file__).name]

    if not test_files:
        console.print("[yellow]No test files found in this module![/yellow]")
        return {'status': 'NO_TESTS', 'passed': 0, 'failed': 0}

    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    # Create results table
    table = Table(title="Test Results", box=box.ROUNDED)
    table.add_column("Test File", style="cyan")
    table.add_column("Test Class", style="yellow")
    table.add_column("Test Method", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for test_file in sorted(test_files):
        module_name = test_file.stem

        try:
            # Import test module
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Find test classes
            for class_name in dir(test_module):
                if class_name.startswith("Test"):
                    test_class = getattr(test_module, class_name)

                    # Create instance
                    try:
                        instance = test_class()
                    except Exception as e:
                        table.add_row(
                            module_name,
                            class_name,
                            "initialization",
                            "[red]❌ ERROR[/red]",
                            "-"
                        )
                        total_failed += 1
                        continue

                    # Run test methods
                    for method_name in dir(instance):
                        if method_name.startswith("test_"):
                            method = getattr(instance, method_name)

                            # Skip template placeholder tests
                            if "pass" in str(method.__code__.co_code):
                                continue

                            # Run test
                            start = time.time()
                            try:
                                method()
                                status = "[green]✅ PASS[/green]"
                                total_passed += 1
                            except AssertionError as e:
                                status = "[red]❌ FAIL[/red]"
                                total_failed += 1
                            except ImportError:
                                status = "[yellow]⏭️ SKIP[/yellow]"
                                total_skipped += 1
                            except Exception as e:
                                status = "[red]💥 ERROR[/red]"
                                total_failed += 1

                            duration = time.time() - start

                            table.add_row(
                                module_name,
                                class_name,
                                method_name,
                                status,
                                f"{duration:.3f}s"
                            )
        except Exception as e:
            console.print(f"[red]Error loading test file {test_file}: {e}[/red]")
            total_failed += 1

    if total_passed + total_failed + total_skipped > 0:
        console.print(table)

        # Summary
        console.print(f"\n📊 Summary:")
        console.print(f"  • Total: {total_passed + total_failed + total_skipped} tests")
        console.print(f"  • ✅ Passed: {total_passed}")
        console.print(f"  • ❌ Failed: {total_failed}")
        if total_skipped > 0:
            console.print(f"  • ⏭️  Skipped: {total_skipped}")

        # Final status
        if total_failed == 0:
            console.print("\n[green bold]✅ All tests passed![/green bold]")
            return {'status': 'PASSED', 'passed': total_passed, 'failed': 0}
        else:
            console.print("\n[red]❌ Some tests failed![/red]")
            return {'status': 'FAILED', 'passed': total_passed, 'failed': total_failed}
    else:
        console.print("[yellow]No actual tests implemented yet (only templates).[/yellow]")
        return {'status': 'NO_TESTS', 'passed': 0, 'failed': 0}


if __name__ == "__main__":
    results = run_module_tests()
    sys.exit(0 if results['status'] == 'PASSED' else 1)
