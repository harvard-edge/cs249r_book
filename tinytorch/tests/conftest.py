"""
Pytest configuration for TinyTorch tests.

This file is automatically loaded by pytest and sets up the test environment.
It also provides a Rich-based educational test output that helps students
understand what each test does and why it matters.

CRITICAL: This conftest validates that the tinytorch package is properly
exported before any tests run. If exports are missing, tests fail fast
with a clear error message.
"""

import sys
import os
import re
from pathlib import Path
from typing import Optional

import pytest

# Add tests directory to Python path so test_utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Add project root to Python path
project_root = tests_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set quiet mode for tinytorch imports during tests
os.environ['TINYTORCH_QUIET'] = '1'


# =============================================================================
# CRITICAL: Package Export Validation
# =============================================================================
# This runs BEFORE any tests to ensure the package is properly built.
# Without this, tests would silently pass because imports return None.

def _validate_package_exported():
    """
    Validate that tinytorch package is properly exported.

    This prevents a critical bug where tests pass because:
    1. tinytorch/__init__.py uses try/except for imports
    2. Missing exports result in Tensor = None (not ImportError)
    3. Tests import None and may pass vacuously

    Returns tuple: (is_valid, error_message)
    """
    errors = []

    # Check 1: Core module files exist
    core_dir = project_root / "tinytorch" / "core"
    required_modules = [
        "tensor.py",
        "activations.py",
        "layers.py",
        "losses.py",
    ]

    for module in required_modules:
        module_path = core_dir / module
        if not module_path.exists():
            errors.append(f"Missing: tinytorch/core/{module}")

    # Check 2: Tensor class is actually importable (not None)
    try:
        from tinytorch import Tensor
        if Tensor is None:
            errors.append("Tensor is None (import failed silently)")
    except ImportError as e:
        errors.append(f"Cannot import Tensor: {e}")

    # Check 3: Verify Tensor is actually the class, not a stub
    try:
        from tinytorch import Tensor
        if Tensor is not None:
            # Try to instantiate - this catches incomplete implementations
            t = Tensor([1, 2, 3])
            if not hasattr(t, 'data'):
                errors.append("Tensor missing 'data' attribute")
            if not hasattr(t, 'shape'):
                errors.append("Tensor missing 'shape' attribute")
    except Exception as e:
        errors.append(f"Tensor instantiation failed: {e}")

    if errors:
        return False, errors
    return True, []


def pytest_configure(config):
    """Configure pytest with TinyTorch-specific settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "module(name): mark test as belonging to a specific module"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )

    # CRITICAL: Validate package is exported before running tests
    # Skip validation if explicitly disabled (e.g., for export tests)
    if os.environ.get('TINYTORCH_SKIP_EXPORT_CHECK') != '1':
        is_valid, errors = _validate_package_exported()
        if not is_valid:
            error_msg = "\n".join(f"  • {e}" for e in errors)
            raise pytest.UsageError(
                f"\n\n"
                f"{'='*70}\n"
                f"❌ TINYTORCH PACKAGE NOT EXPORTED\n"
                f"{'='*70}\n\n"
                f"The tinytorch package is not properly built. Tests cannot run.\n\n"
                f"Errors found:\n{error_msg}\n\n"
                f"To fix this, run:\n\n"
                f"    tito dev export --all\n\n"
                f"This exports all module notebooks to the tinytorch package.\n"
                f"{'='*70}\n"
            )

# Import test utilities to make them available
try:
    from test_utils import setup_integration_test, create_test_tensor, assert_tensors_close
except ImportError:
    pass  # test_utils not yet created or has issues

# Register plugins (note: pytest_tinytorch was removed during test cleanup)


# =============================================================================
# Educational Test Output Plugin
# =============================================================================

def extract_test_purpose(docstring: Optional[str]) -> dict:
    """
    Extract WHAT/WHY/HOW from test docstrings.

    Returns dict with keys: 'what', 'why', 'learning', 'raw'
    """
    if not docstring:
        return {'what': None, 'why': None, 'learning': None, 'raw': None}

    result = {'raw': docstring.strip()}

    # Extract WHAT section
    what_match = re.search(r'WHAT:\s*(.+?)(?=\n\s*\n|WHY:|$)', docstring, re.DOTALL | re.IGNORECASE)
    if what_match:
        result['what'] = what_match.group(1).strip()

    # Extract WHY section
    why_match = re.search(r'WHY:\s*(.+?)(?=\n\s*\n|STUDENT|HOW:|$)', docstring, re.DOTALL | re.IGNORECASE)
    if why_match:
        result['why'] = why_match.group(1).strip()

    # Extract STUDENT LEARNING section
    learning_match = re.search(r'STUDENT LEARNING:\s*(.+?)(?=\n\s*\n|$)', docstring, re.DOTALL | re.IGNORECASE)
    if learning_match:
        result['learning'] = learning_match.group(1).strip()

    return result


def get_module_from_path(path: str) -> Optional[str]:
    """Extract module number from test file path."""
    match = re.search(r'/(\d{2})_(\w+)/', str(path))
    if match:
        return f"Module {match.group(1)}: {match.group(2).title()}"
    return None


class TinyTorchTestReporter:
    """Rich-based test reporter for educational output."""

    def __init__(self):
        self.current_module = None
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.use_rich = False

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            self.console = Console()
            self.use_rich = True
        except ImportError:
            self.console = None

    def print_test_start(self, nodeid: str, docstring: Optional[str]):
        """Print when a test starts (only in verbose mode)."""
        if not self.use_rich:
            return

        # Extract test name
        parts = nodeid.split("::")
        test_name = parts[-1] if parts else nodeid

        # Get module info
        module = get_module_from_path(nodeid)
        if module and module != self.current_module:
            self.current_module = module
            self.console.print(f"\n[bold blue]━━━ {module} ━━━[/bold blue]")

        # Get purpose from docstring
        purpose = extract_test_purpose(docstring)
        what = purpose.get('what')

        if what:
            # Truncate to first line/sentence
            what_short = what.split('\n')[0][:60]
            self.console.print(f"  [dim]⏳[/dim] {test_name}: {what_short}...")
        else:
            self.console.print(f"  [dim]⏳[/dim] {test_name}...")

    def print_test_result(self, nodeid: str, outcome: str, docstring: Optional[str] = None,
                          longrepr=None):
        """Print test result with educational context."""
        if not self.use_rich:
            return

        parts = nodeid.split("::")
        test_name = parts[-1] if parts else nodeid

        if outcome == "passed":
            self.passed += 1
            self.console.print(f"  [green]✓[/green] {test_name}")
        elif outcome == "skipped":
            self.skipped += 1
            self.console.print(f"  [yellow]⊘[/yellow] {test_name} [dim](skipped)[/dim]")
        elif outcome == "failed":
            self.failed += 1
            self.console.print(f"  [red]✗[/red] {test_name}")

            # Show educational context on failure
            purpose = extract_test_purpose(docstring)
            if purpose.get('what') or purpose.get('why'):
                from rich.panel import Panel
                from rich.text import Text

                content = Text()
                if purpose.get('what'):
                    content.append("WHAT: ", style="bold cyan")
                    content.append(purpose['what'][:200] + "\n\n")
                if purpose.get('why'):
                    content.append("WHY THIS MATTERS: ", style="bold yellow")
                    content.append(purpose['why'][:300])

                self.console.print(Panel(content, title="[red]Test Failed[/red]",
                                        border_style="red", padding=(0, 1)))

    def print_summary(self):
        """Print final summary."""
        if not self.use_rich:
            return

        total = self.passed + self.failed + self.skipped

        self.console.print("\n" + "━" * 50)
        status = "[green]ALL PASSED[/green]" if self.failed == 0 else f"[red]{self.failed} FAILED[/red]"
        self.console.print(f"[bold]{status}[/bold] | {self.passed} passed, {self.skipped} skipped, {total} total")


# Global reporter instance
_reporter = TinyTorchTestReporter()


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_collection_modifyitems(session, config, items):
    """Modify test collection to add educational metadata."""
    for item in items:
        # Auto-detect module from path
        module = get_module_from_path(str(item.fspath))
        if module:
            # Store module info for later use
            item._tinytorch_module = module


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for educational output."""
    outcome = yield
    report = outcome.get_result()

    # Only process the "call" phase (not setup/teardown)
    if report.when == "call":
        # Get docstring from test function
        docstring = item.function.__doc__ if hasattr(item, 'function') else None

        # Store for later use if needed
        report._tinytorch_docstring = docstring


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add educational summary at the end of test run."""
    # Check if we should show educational summary
    if hasattr(config, '_tinytorch_show_summary') and config._tinytorch_show_summary:
        _reporter.print_summary()


# =============================================================================
# Custom Test Runner Command (for tito test)
# =============================================================================

def run_tests_with_rich_output(test_path: str = None, verbose: bool = True):
    """
    Run tests with Rich educational output.

    This can be called from tito CLI to provide a better student experience.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Header
    console.print(Panel(
        "[bold]🧪 TinyTorch Test Runner[/bold]\n"
        "Running tests with educational context...",
        border_style="blue"
    ))

    # Build pytest args
    args = ["-v", "--tb=short"]
    if test_path:
        args.append(test_path)

    # Run pytest
    exit_code = pytest.main(args)

    return exit_code
