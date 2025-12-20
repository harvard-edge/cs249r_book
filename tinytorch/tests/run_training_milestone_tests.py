#!/usr/bin/env python3
"""
Test Runner for Training Milestone (Modules 01-08)
===================================================

Runs all tests for the core modules needed for the training milestone:
  01_tensor      - Data structure
  02_activations - Activation functions
  03_layers      - Neural network layers
  04_losses      - Loss functions
  05_dataloader  - Data loading and batching
  06_autograd    - Automatic differentiation
  07_optimizers  - SGD, Adam, etc.
  08_training    - Training loops

This script runs all tests and provides a comprehensive summary.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Define modules to test
MODULES = [
    ("01_tensor", "Tensor"),
    ("02_activations", "Activations"),
    ("03_layers", "Layers"),
    ("04_losses", "Losses"),
    ("05_dataloader", "DataLoader"),
    ("06_autograd", "Autograd"),
    ("07_optimizers", "Optimizers"),
    ("08_training", "Training"),
]

def run_module_tests(module_dir: str) -> dict:
    """Run tests for a single module and return results."""
    test_path = Path(__file__).parent / module_dir / "run_all_tests.py"

    if not test_path.exists():
        return {
            "status": "skip",
            "passed": 0,
            "failed": 0,
            "total": 0,
            "error": "Test file not found"
        }

    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=test_path.parent,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Parse output to extract test counts
        output = result.stdout + result.stderr

        # Strip ANSI codes for easier parsing
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_output = ansi_escape.sub('', output)

        # Look for summary lines
        passed = failed = total = 0
        for line in clean_output.split('\n'):
            if '• Total:' in line or 'Total:' in line:
                try:
                    total = int(''.join(filter(str.isdigit, line.split('Total:')[1].split()[0])))
                except:
                    pass
            if '✅ Passed:' in line or 'Passed:' in line:
                try:
                    passed = int(''.join(filter(str.isdigit, line.split('Passed:')[1].split()[0])))
                except:
                    pass
            if '❌ Failed:' in line or 'Failed:' in line:
                try:
                    failed = int(''.join(filter(str.isdigit, line.split('Failed:')[1].split()[0])))
                except:
                    pass

        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "passed": passed,
            "failed": failed,
            "total": total if total > 0 else passed + failed,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "passed": 0,
            "failed": 0,
            "total": 0,
            "error": "Timeout (>60s)"
        }
    except Exception as e:
        return {
            "status": "error",
            "passed": 0,
            "failed": 0,
            "total": 0,
            "error": str(e)
        }

def main():
    console.print(Panel.fit(
        "[bold cyan]Training Milestone Test Suite[/bold cyan]\n"
        "[dim]Testing modules 01-08 for training readiness[/dim]",
        border_style="cyan"
    ))

    results = {}

    # Run tests for each module
    for module_dir, module_name in MODULES:
        console.print(f"\n[bold]Testing {module_name}[/bold] ({module_dir})...")
        results[module_dir] = run_module_tests(module_dir)

        # Show quick status
        status = results[module_dir]["status"]
        if status == "pass":
            emoji = "✅"
            color = "green"
        elif status == "fail":
            emoji = "❌"
            color = "red"
        elif status == "skip":
            emoji = "⏭️"
            color = "yellow"
        else:
            emoji = "💥"
            color = "red"

        passed = results[module_dir]["passed"]
        total = results[module_dir]["total"]
        console.print(f"  {emoji} [{color}]{passed}/{total} tests passed[/{color}]")

    # Create summary table
    console.print("\n")
    table = Table(title="Test Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("Module", style="cyan", width=20)
    table.add_column("Name", style="white", width=15)
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Total", justify="right", style="blue")
    table.add_column("Pass Rate", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    total_passed = 0
    total_failed = 0
    total_tests = 0

    for module_dir, module_name in MODULES:
        result = results[module_dir]
        passed = result["passed"]
        failed = result["failed"]
        total = result["total"]

        total_passed += passed
        total_failed += failed
        total_tests += total

        if total > 0:
            pass_rate = f"{(passed/total)*100:.0f}%"
        else:
            pass_rate = "N/A"

        if result["status"] == "pass":
            status = "✅ PASS"
        elif result["status"] == "fail":
            status = "❌ FAIL"
        elif result["status"] == "skip":
            status = "⏭️  SKIP"
        else:
            status = "💥 ERROR"

        table.add_row(
            module_dir,
            module_name,
            str(passed),
            str(failed),
            str(total),
            pass_rate,
            status
        )

    # Add totals row
    if total_tests > 0:
        overall_pass_rate = f"{(total_passed/total_tests)*100:.1f}%"
    else:
        overall_pass_rate = "N/A"

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "[bold]All Modules[/bold]",
        f"[bold green]{total_passed}[/bold green]",
        f"[bold red]{total_failed}[/bold red]",
        f"[bold blue]{total_tests}[/bold blue]",
        f"[bold yellow]{overall_pass_rate}[/bold yellow]",
        ""
    )

    console.print(table)

    # Final assessment
    console.print("\n")
    if total_failed == 0 and total_tests > 0:
        console.print(Panel.fit(
            "[bold green]🎉 ALL TESTS PASSED![/bold green]\n"
            f"All {total_tests} tests across modules 01-08 are passing.\n"
            "Training milestone is ready!",
            border_style="green"
        ))
        sys.exit(0)
    elif total_passed / total_tests >= 0.9:
        console.print(Panel.fit(
            f"[bold yellow]⚠️  MOSTLY PASSING ({overall_pass_rate})[/bold yellow]\n"
            f"{total_passed}/{total_tests} tests passing.\n"
            "Some fixes needed but close to ready.",
            border_style="yellow"
        ))
        sys.exit(1)
    else:
        console.print(Panel.fit(
            f"[bold red]❌ TESTS FAILING ({overall_pass_rate})[/bold red]\n"
            f"Only {total_passed}/{total_tests} tests passing.\n"
            "Significant work needed before training milestone is ready.",
            border_style="red"
        ))
        sys.exit(1)

if __name__ == "__main__":
    main()
