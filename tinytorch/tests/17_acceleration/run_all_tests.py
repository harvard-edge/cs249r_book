#!/usr/bin/env python3
"""
Run all tests for Module 17: Acceleration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_module_tests():
    """Run all tests for Module 17: Acceleration."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel("[bold blue]Module 17: Acceleration - Test Suite[/bold blue]", expand=False))

    test_files = list(Path(__file__).parent.glob("test_*.py"))

    if not test_files:
        console.print("[yellow]No test files found - tests not yet implemented[/yellow]")
        return {'status': 'NO_TESTS', 'passed': 0, 'failed': 0}

    console.print(f"[green]Found {len(test_files)} test files[/green]")
    console.print("[dim]Test implementation pending...[/dim]")

    return {'status': 'PENDING', 'passed': 0, 'failed': 0}

if __name__ == "__main__":
    result = run_module_tests()
    sys.exit(0 if result['status'] in ['SUCCESS', 'NO_TESTS', 'PENDING'] else 1)
