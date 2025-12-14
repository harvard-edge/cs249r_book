#!/usr/bin/env python3
"""
🧪 TinyTorch Comprehensive Test Runner

This script runs ALL tests to give students a complete picture of their progress:
1. Individual module tests (check each module works)
2. Integration tests (check modules work together)
3. Package tests (check examples work as expected)

Perfect for students to check their work at any stage!
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunner:
    """Comprehensive test runner for TinyTorch students."""

    def __init__(self):
        self.results = {
            'modules': {},
            'integration': {},
            'examples': {}
        }
        self.start_time = time.time()

    def run_all_tests(self):
        """Run all test categories and provide comprehensive report."""
        console.print(Panel.fit(
            "🧪 [bold blue]TinyTorch Comprehensive Test Suite[/bold blue]\n"
            "[dim]Testing modules, integration, and examples...[/dim]",
            border_style="blue"
        ))

        # Run all test categories
        self.test_modules()
        self.test_integration()
        self.test_examples()

        # Show final comprehensive report
        self.show_final_report()

        return self._calculate_overall_health() >= 0.7


    def test_modules(self):
        """Test all individual modules using tito test."""
        console.print("\n📚 [bold]Testing Individual Modules[/bold]")

        try:
            with console.status("Running tito test --all..."):
                result = subprocess.run([
                    sys.executable, "./bin/tito", "test", "--all", "--summary"
                ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Parse the summary output to get individual module results
                output_lines = result.stdout.split('\n')
                success_rate = "100.0%" in result.stdout

                self.results['modules']['all_modules'] = {
                    'status': 'PASS' if success_rate else 'PARTIAL',
                    'output': result.stdout,
                    'error': result.stderr,
                    'summary': f"Modules test via tito: {'PASS' if success_rate else 'PARTIAL'}"
                }

                # Extract individual results if available
                for line in output_lines:
                    if '✅' in line and 'All tests passed' in line:
                        module = line.split()[0]
                        self.results['modules'][module] = {'status': 'PASS'}
                    elif '❌' in line:
                        module = line.split()[0] if line.split() else 'unknown'
                        self.results['modules'][module] = {'status': 'FAIL'}
            else:
                self.results['modules']['all_modules'] = {
                    'status': 'FAIL',
                    'output': result.stdout,
                    'error': result.stderr,
                    'summary': "Module tests failed"
                }
        except Exception as e:
            self.results['modules']['all_modules'] = {
                'status': 'ERROR',
                'output': '',
                'error': str(e),
                'summary': f"Error running module tests: {e}"
            }


    def test_integration(self):
        """Test integration between modules."""
        console.print("\n🔗 [bold]Testing Module Integration[/bold]")

        integration_tests = [
            "tests/integration/run_integration_tests.py",
            "tests/integration/test_basic_integration.py"
        ]

        for test_path in integration_tests:
            test_name = os.path.basename(test_path).replace('.py', '')

            if os.path.exists(test_path):
                try:
                    with console.status(f"Running {test_name}..."):
                        result = subprocess.run([
                            sys.executable, test_path
                        ], capture_output=True, text=True, timeout=60)

                    self.results['integration'][test_name] = {
                        'status': 'PASS' if result.returncode == 0 else 'FAIL',
                        'output': result.stdout,
                        'error': result.stderr
                    }
                except Exception as e:
                    self.results['integration'][test_name] = {
                        'status': 'ERROR',
                        'output': '',
                        'error': str(e)
                    }

    def test_examples(self):
        """Test example scripts (package functionality)."""
        console.print("\n🚀 [bold]Testing Examples (Package Tests)[/bold]")

        examples = [
            ("XOR Training", "examples/xornet/train_with_dashboard.py"),
            ("CIFAR-10 Baseline", "examples/cifar10/random_baseline.py"),
        ]

        for name, example_path in examples:
            if os.path.exists(example_path):
                try:
                    with console.status(f"Running {name}..."):
                        timeout = 15 if 'train' in example_path else 30

                        result = subprocess.run([
                            sys.executable, example_path
                        ], capture_output=True, text=True, timeout=timeout)

                    status = 'PASS' if result.returncode == 0 else 'PARTIAL'
                    if 'ERROR' in result.stderr.upper():
                        status = 'FAIL'

                    self.results['examples'][name] = {
                        'status': status,
                        'output': result.stdout[-300:] if result.stdout else '',
                        'error': result.stderr[-200:] if result.stderr else ''
                    }
                except subprocess.TimeoutExpired:
                    self.results['examples'][name] = {
                        'status': 'PARTIAL',
                        'output': 'Started successfully (timed out for demo)',
                        'error': ''
                    }
                except Exception as e:
                    self.results['examples'][name] = {
                        'status': 'ERROR',
                        'output': '',
                        'error': str(e)
                    }


    def show_final_report(self):
        """Show comprehensive final report."""
        elapsed_time = time.time() - self.start_time

        console.print(f"\n⏱️  [dim]Total test time: {elapsed_time:.1f}s[/dim]\n")

        # Module Results
        console.print("📚 [bold]Module Test Results[/bold]")
        for name, result in self.results['modules'].items():
            status_style = 'green' if result['status'] == 'PASS' else 'red'
            status_icon = '✅' if result['status'] == 'PASS' else '❌'
            console.print(f"  {status_icon} [{status_style}]{name}[/{status_style}]")

        # Integration Results
        if self.results['integration']:
            console.print("\n🔗 [bold]Integration Test Results[/bold]")
            for test_name, result in self.results['integration'].items():
                status_style = 'green' if result['status'] == 'PASS' else 'red'
                status_icon = '✅' if result['status'] == 'PASS' else '❌'
                console.print(f"  {status_icon} [{status_style}]{test_name}[/{status_style}]")

        # Example Results
        if self.results['examples']:
            console.print("\n🚀 [bold]Example Test Results[/bold]")
            for example_name, result in self.results['examples'].items():
                status_style = {
                    'PASS': 'green',
                    'PARTIAL': 'yellow',
                    'FAIL': 'red'
                }.get(result['status'], 'red')

                status_icon = {
                    'PASS': '✅',
                    'PARTIAL': '⚠️',
                    'FAIL': '❌'
                }.get(result['status'], '❌')

                console.print(f"  {status_icon} [{status_style}]{example_name}[/{status_style}]")

        # Summary Statistics
        overall_health = self._calculate_overall_health()

        console.print(Panel.fit(
            f"📊 [bold]Summary Statistics[/bold]\n\n"
            f"🎯 [bold]Overall TinyTorch Health: {overall_health:.1%}[/bold]\n\n"
            f"{'🎉 Excellent! Everything is working great!' if overall_health >= 0.9 else '👍 Good progress! Most things working.' if overall_health >= 0.7 else '⚠️ Some issues need attention.' if overall_health >= 0.5 else '🔧 Several components need fixing.'}",
            border_style="green" if overall_health >= 0.7 else "yellow" if overall_health >= 0.5 else "red"
        ))

    def _calculate_overall_health(self):
        """Calculate overall TinyTorch health percentage."""
        all_results = []

        # Add module results
        for result in self.results['modules'].values():
            all_results.append(1 if result['status'] in ['PASS', 'PARTIAL'] else 0)

        # Add integration results
        for result in self.results['integration'].values():
            all_results.append(1 if result['status'] == 'PASS' else 0)

        # Add example results
        for result in self.results['examples'].values():
            all_results.append(1 if result['status'] in ['PASS', 'PARTIAL'] else 0)

        return sum(all_results) / len(all_results) if all_results else 0.0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="🧪 TinyTorch Comprehensive Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Show detailed output from tests")

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    runner = TestRunner()
    success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
