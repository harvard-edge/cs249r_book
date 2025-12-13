"""
TinyTorch Educational Test Plugin for Pytest
=============================================

This plugin provides Rich-formatted output that helps students understand
what tests are checking and why they matter.

USAGE:
    pytest --tinytorch      # Enable educational output
    pytest --tinytorch -v   # Verbose educational output

Or run through tito:
    tito test --edu         # Educational mode
"""

import re
from typing import Optional, Dict, Any
import pytest


def pytest_addoption(parser):
    """Add TinyTorch-specific command line options."""
    group = parser.getgroup('tinytorch', 'TinyTorch educational testing')
    group.addoption(
        '--tinytorch',
        action='store_true',
        dest='tinytorch_edu',
        default=False,
        help='Enable TinyTorch educational test output'
    )


def pytest_configure(config):
    """Configure the plugin."""
    if config.getoption('tinytorch_edu', False):
        config.pluginmanager.register(TinyTorchReporter(config), 'tinytorch_reporter')


class TinyTorchReporter:
    """
    Rich-based reporter that shows educational context for tests.

    Features:
    - Module grouping with descriptions
    - WHAT/WHY extraction from docstrings
    - Clear pass/fail indicators
    - Educational failure messages
    """

    def __init__(self, config):
        self.config = config
        self.current_module = None
        self.stats = {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0}
        self.failures = []

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False

    def _extract_purpose(self, docstring: Optional[str]) -> Dict[str, Optional[str]]:
        """Extract WHAT/WHY/LEARNING from docstring."""
        if not docstring:
            return {'what': None, 'why': None, 'learning': None}

        result = {}

        # Extract WHAT
        what_match = re.search(r'WHAT:\s*(.+?)(?=\n\s*\n|WHY:|$)', docstring, re.DOTALL | re.IGNORECASE)
        result['what'] = what_match.group(1).strip() if what_match else None

        # Extract WHY
        why_match = re.search(r'WHY:\s*(.+?)(?=\n\s*\n|STUDENT|HOW:|$)', docstring, re.DOTALL | re.IGNORECASE)
        result['why'] = why_match.group(1).strip() if why_match else None

        # Extract STUDENT LEARNING
        learning_match = re.search(r'STUDENT LEARNING:\s*(.+?)(?=\n\s*\n|$)', docstring, re.DOTALL)
        result['learning'] = learning_match.group(1).strip() if learning_match else None

        return result

    def _get_module_info(self, nodeid: str) -> Optional[str]:
        """Extract module name from test path."""
        match = re.search(r'/(\d{2})_(\w+)/', nodeid)
        if match:
            num, name = match.groups()
            return f"Module {num}: {name.replace('_', ' ').title()}"

        # Check for other test categories
        if '/integration/' in nodeid:
            return "Integration Tests"
        if '/regression/' in nodeid:
            return "Regression Tests"
        if '/e2e/' in nodeid:
            return "End-to-End Tests"

        return None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection_finish(self, session):
        """Called after collection, show what we're testing."""
        yield

        if not self.rich_available:
            return

        from rich.panel import Panel
        from rich.table import Table

        # Group tests by module
        modules = {}
        for item in session.items:
            module = self._get_module_info(item.nodeid) or "Other Tests"
            if module not in modules:
                modules[module] = []
            modules[module].append(item.name)

        # Create summary table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Module", style="cyan")
        table.add_column("Tests", justify="right")
        table.add_column("Sample Tests", style="dim")

        for module, tests in sorted(modules.items()):
            sample = ", ".join(tests[:2])
            if len(tests) > 2:
                sample += f", ... (+{len(tests)-2} more)"
            table.add_row(module, str(len(tests)), sample)

        self.console.print(Panel(
            table,
            title="[bold]🧪 TinyTorch Test Suite[/bold]",
            subtitle=f"[dim]{len(session.items)} tests to run[/dim]",
            border_style="blue"
        ))
        self.console.print()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item):
        """Called for each test."""
        # Check if we're entering a new module
        module = self._get_module_info(item.nodeid)
        if self.rich_available and module and module != self.current_module:
            self.current_module = module
            self.console.print(f"\n[bold blue]━━━ {module} ━━━[/bold blue]")

        yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Process test results."""
        outcome = yield
        report = outcome.get_result()

        if report.when != "call":
            return

        if not self.rich_available:
            return

        # Get test info
        test_name = item.name
        docstring = item.function.__doc__ if hasattr(item, 'function') else None
        purpose = self._extract_purpose(docstring)

        # Format output based on result
        if report.passed:
            self.stats['passed'] += 1
            what = purpose.get('what', '')
            if what:
                what_short = what.split('\n')[0][:50]
                self.console.print(f"  [green]✓[/green] {test_name} [dim]- {what_short}[/dim]")
            else:
                self.console.print(f"  [green]✓[/green] {test_name}")

        elif report.skipped:
            self.stats['skipped'] += 1
            self.console.print(f"  [yellow]⊘[/yellow] {test_name} [dim](skipped)[/dim]")

        elif report.failed:
            self.stats['failed'] += 1
            self.console.print(f"  [red]✗[/red] {test_name}")

            # Store failure info for detailed output
            self.failures.append({
                'name': test_name,
                'nodeid': item.nodeid,
                'purpose': purpose,
                'longrepr': report.longreprtext
            })

    def pytest_sessionfinish(self, session, exitstatus):
        """Called at the end of the session."""
        if not self.rich_available:
            return

        from rich.panel import Panel
        from rich.text import Text

        self.console.print()

        # Show failure details with educational context
        if self.failures:
            self.console.print("[bold red]━━━ Failed Tests ━━━[/bold red]\n")

            for failure in self.failures:
                # Create educational failure panel
                content = Text()

                purpose = failure['purpose']
                if purpose.get('what'):
                    content.append("📋 WHAT: ", style="bold cyan")
                    content.append(purpose['what'][:200] + "\n\n", style="white")

                if purpose.get('why'):
                    content.append("❓ WHY: ", style="bold yellow")
                    content.append(purpose['why'][:300] + "\n\n", style="white")

                if purpose.get('learning'):
                    content.append("💡 TIP: ", style="bold green")
                    content.append(purpose['learning'][:200] + "\n\n", style="white")

                # Add error excerpt
                error_lines = failure['longrepr'].split('\n')
                error_excerpt = '\n'.join(error_lines[-10:])  # Last 10 lines
                content.append("🔍 Error:\n", style="bold red")
                content.append(error_excerpt[:500], style="dim")

                self.console.print(Panel(
                    content,
                    title=f"[red]✗ {failure['name']}[/red]",
                    border_style="red",
                    padding=(1, 2)
                ))
                self.console.print()

        # Summary
        total = sum(self.stats.values())
        passed = self.stats['passed']
        failed = self.stats['failed']
        skipped = self.stats['skipped']

        if failed == 0:
            status_style = "green"
            status_text = "ALL TESTS PASSED"
            emoji = "🎉"
        else:
            status_style = "red"
            status_text = f"{failed} TESTS FAILED"
            emoji = "❌"

        summary = Text()
        summary.append(f"\n{emoji} ", style="bold")
        summary.append(status_text, style=f"bold {status_style}")
        summary.append(f"\n\n   Passed: {passed}", style="green")
        summary.append(f"   Failed: {failed}", style="red")
        summary.append(f"   Skipped: {skipped}", style="yellow")
        summary.append(f"   Total: {total}", style="dim")

        self.console.print(Panel(summary, border_style=status_style))
