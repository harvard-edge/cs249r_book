#!/usr/bin/env python3
"""
Run all tests for Module 04: Dense/Networks
"""

import sys
from pathlib import Path
import importlib
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

    # Header
    console.print(Panel("[bold blue]Module 04: Dense/Networks - Test Suite[/bold blue]",
                       expand=False))

    # Find all test files in this module
    test_files = list(Path(__file__).parent.glob("test_*.py"))
    test_files = [f for f in test_files if f.name != Path(__file__).name]

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


if __name__ == "__main__":
    results = run_module_tests()
    sys.exit(0 if results['status'] == 'PASSED' else 1)
