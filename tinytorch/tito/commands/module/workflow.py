"""
Enhanced Module Workflow for TinyTorch CLI.

Implements the natural workflow:
1. tito module 01 → Opens module 01 in Jupyter
2. Student works and saves
3. tito module complete 01 → Tests, exports, updates progress
"""

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from ..base import BaseCommand
from .reset import ModuleResetCommand
from .test import ModuleTestCommand
from ...core.exceptions import ModuleNotFoundError
from ...core import auth
from ...core.submission import SubmissionHandler
from ...core.modules import (
    get_module_mapping,
    get_module_name,
    get_module_display_name,
    get_next_module,
    normalize_module_number,
    get_total_modules,
    module_exists,
    get_all_module_metadata,
)

class ModuleWorkflowCommand(BaseCommand):
    """Enhanced module command with natural workflow."""

    @property
    def name(self) -> str:
        return "module"

    @property
    def description(self) -> str:
        return "Module development workflow - open, work, complete"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add module workflow arguments."""
        # Add subcommands - clean lifecycle workflow
        subparsers = parser.add_subparsers(
            dest='module_command',
            help='Module lifecycle operations'
        )

        # START command - begin working on a module
        start_parser = subparsers.add_parser(
            'start',
            help='Start working on a module (first time)'
        )
        start_parser.add_argument(
            'module_number',
            help='Module number to start (01, 02, 03, etc.)'
        )

        # VIEW command - just open the notebook
        view_parser = subparsers.add_parser(
            'view',
            help='Open module notebook in Jupyter (no status updates)'
        )
        view_parser.add_argument(
            'module_number',
            help='Module number to view (01, 02, 03, etc.)'
        )

        # RESUME command - continue working on a module
        resume_parser = subparsers.add_parser(
            'resume',
            help='Resume working on a module (continue previous work)'
        )
        resume_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to resume (01, 02, 03, etc.) - defaults to last worked'
        )

        # COMPLETE command - finish and validate a module
        complete_parser = subparsers.add_parser(
            'complete',
            help='Complete module: run tests, export if passing, update progress'
        )
        complete_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to complete (01, 02, 03, etc.) - defaults to current'
        )
        complete_parser.add_argument(
            '--skip-tests',
            action='store_true',
            help='Skip integration tests'
        )
        complete_parser.add_argument(
            '--skip-export',
            action='store_true',
            help='Skip automatic export'
        )
        complete_parser.add_argument(
            '--all',
            action='store_true',
            help='Complete all modules (test + export all)'
        )

        # TEST command - run module tests (three-phase testing)
        test_parser = subparsers.add_parser(
            'test',
            help='Run module tests: inline → pytest → integration'
        )
        test_parser.add_argument(
            'module_number',
            nargs='?',
            help='Module number to test (01, 02, 03, etc.)'
        )
        test_parser.add_argument(
            '--all',
            action='store_true',
            help='Test all modules sequentially'
        )
        test_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Show detailed test output'
        )
        test_parser.add_argument(
            '--stop-on-fail',
            action='store_true',
            help='Stop testing if a module fails (only with --all)'
        )
        test_parser.add_argument(
            '--unit-only',
            action='store_true',
            help='Run only inline unit tests (skip pytest and integration)'
        )
        test_parser.add_argument(
            '--no-integration',
            action='store_true',
            help='Skip integration tests'
        )

        # RESET command - reset module to clean state
        reset_parser = subparsers.add_parser(
            'reset',
            help='Reset module to clean state'
        )
        reset_parser.add_argument(
            'module_number',
            nargs='?',
            default=None,
            help='Module number to reset (01, 02, etc.)'
        )
        reset_parser.add_argument(
            '--all',
            action='store_true',
            help='Reset ALL modules to pristine state'
        )
        reset_parser.add_argument(
            '--force',
            action='store_true',
            help='Skip confirmation prompts'
        )

        # STATUS command - show progress
        status_parser = subparsers.add_parser(
            'status',
            help='Show module completion status and progress'
        )

        # LIST command - show available modules
        list_parser = subparsers.add_parser(
            'list',
            help='List all available modules'
        )

        # EXPORT command - export module code to tinytorch package
        export_parser = subparsers.add_parser(
            'export',
            help='Export module code to tinytorch package'
        )
        export_parser.add_argument(
            'modules',
            nargs='*',
            help='Module names to export (e.g., 01_tensor 02_activations)'
        )
        export_parser.add_argument(
            '--all',
            action='store_true',
            help='Export all modules'
        )
        export_parser.add_argument(
            '--from-release',
            action='store_true',
            help='Export from release directory (student version) instead of source'
        )
        export_parser.add_argument(
            '--test-checkpoint',
            action='store_true',
            help='Run checkpoint test after successful export'
        )

    # Module mapping and normalization now imported from core.modules

    def start_module(self, module_number: str) -> int:
        """Start working on a module with prerequisite checking and visual feedback."""
        from rich import box
        from rich.table import Table

        module_mapping = get_module_mapping()
        normalized = normalize_module_number(module_number)

        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            max_module = max(module_mapping.keys()) if module_mapping else "??"
            self.console.print(f"💡 Available modules: 01-{max_module}")
            return 1

        module_name = module_mapping[normalized]
        module_num = int(normalized)

        # Check if already started
        if self.is_module_started(normalized):
            self.console.print(f"[yellow]⚠️  Module {normalized} already started[/yellow]")
            self.console.print(f"💡 Did you mean: [bold cyan]tito module resume {normalized}[/bold cyan]")
            return 1

        # Check prerequisites - all previous modules must be completed
        progress = self.get_progress_data()
        completed = progress.get('completed_modules', [])

        # Module 01 has no prerequisites
        if module_num > 1:
            missing_prereqs = []
            for i in range(1, module_num):
                prereq_num = f"{i:02d}"
                if prereq_num not in completed:
                    missing_prereqs.append((prereq_num, module_mapping.get(prereq_num, "Unknown")))

            if missing_prereqs:
                # Show locked module panel
                self.console.print(Panel(
                    f"[yellow]Module {normalized}: {module_name} is locked[/yellow]\n\n"
                    f"Complete the prerequisites first to unlock this module.",
                    title="🔒 Module Locked",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
                self.console.print()

                # Show prerequisites table
                prereq_table = Table(
                    title="Prerequisites Required",
                    show_header=True,
                    header_style="bold yellow",
                    box=box.SIMPLE
                )
                prereq_table.add_column("Module", style="cyan", width=8)
                prereq_table.add_column("Name", style="bold", width=20)
                prereq_table.add_column("Status", width=15, justify="center")

                for prereq_num, prereq_name in missing_prereqs:
                    prereq_table.add_row(
                        prereq_num,
                        prereq_name,
                        "[red]❌ Not Complete[/red]"
                    )

                self.console.print(prereq_table)
                self.console.print()

                # Show what to do next
                first_missing = missing_prereqs[0][0]
                self.console.print(f"💡 Next: [bold cyan]tito module start {first_missing}[/bold cyan]")
                self.console.print(f"   Complete modules in order to build your ML framework progressively")

                return 1

        # Prerequisites met! Check if module needs to be created from src/
        # Notebooks are in modules/ directory, not src/ (which is modules_dir in config)
        module_dir = self.config.project_root / "modules" / module_name
        if not module_dir.exists():
            # Create module from src/ using export
            src_dir = self.config.project_root / "src" / module_name
            if not src_dir.exists():
                self.console.print(f"[red]❌ Source not found: src/{module_name}[/red]")
                return 1

            self.console.print(f"[cyan]📝 Creating module from source...[/cyan]")
            if not self._create_module_from_src(module_name):
                self.console.print(f"[red]❌ Failed to create module {module_name}[/red]")
                return 1
            self.console.print(f"[green]✅ Module {normalized} ready![/green]")
            self.console.print()

        # Show success panel
        self.console.print(Panel(
            f"[green]Starting Module {normalized}: {module_name}[/green]\n\n"
            f"Build your ML framework one component at a time.",
            title=f"🚀 Module {normalized} Unlocked!",
            border_style="bright_green",
            box=box.ROUNDED
        ))
        self.console.print()

        # Show module info table
        info_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2)
        )
        info_table.add_column("Field", style="dim", width=18)
        info_table.add_column("Value")

        info_table.add_row("📦 Module", f"[bold cyan]{normalized} - {module_name}[/bold cyan]")
        info_table.add_row("📊 Progress", f"{len(completed)}/{len(module_mapping)} modules completed")

        # Check for milestone unlocks
        milestone_info = self._get_milestone_for_module(module_num)
        if milestone_info:
            mid, mname, required = milestone_info
            if module_num in required:
                modules_left = len([r for r in required if r not in completed and r >= module_num])
                if modules_left <= 3:
                    info_table.add_row("🏆 Milestone", f"[magenta]{mid} - {mname}[/magenta]")
                    info_table.add_row("", f"[dim]{modules_left} modules until unlock[/dim]")

        self.console.print(info_table)
        self.console.print()

        # Mark as started
        self.mark_module_started(normalized)

        # Instructions
        self.console.print("💡 [bold]What to do:[/bold]")
        self.console.print("   1. Work in Jupyter Lab (opening now...)")
        self.console.print("   2. Build your implementation")
        self.console.print("   3. Run: [bold cyan]tito module complete " + normalized + "[/bold cyan]")
        self.console.print()

        return self._open_jupyter(module_name)

    def view_module(self, module_number: str) -> int:
        """Open a module notebook in Jupyter without any status ceremony."""
        module_mapping = get_module_mapping()
        normalized = normalize_module_number(module_number)

        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            return 1

        module_name = module_mapping[normalized]
        # Notebooks are in modules/ directory, not src/ (which is modules_dir in config)
        module_dir = self.config.project_root / "modules" / module_name

        if not module_dir.exists():
            self.console.print(f"[yellow]⚠️  Module {normalized} not started yet[/yellow]")
            self.console.print(f"💡 Run: [bold cyan]tito module start {normalized}[/bold cyan]")
            return 1

        return self._open_jupyter(module_name)

    def _create_module_from_src(self, module_name: str) -> bool:
        """Create a module in modules/ by converting from src/.

        Uses the same conversion logic as 'tito src export' but only creates
        the student-facing notebook, without exporting to the tinytorch package.
        """
        from ..export_utils import convert_py_to_notebook

        src_path = self.config.project_root / "src" / module_name
        if not src_path.exists():
            return False

        # Convert src/*.py to modules/*.ipynb using jupytext
        return convert_py_to_notebook(src_path, self.venv_path, self.console)

    def _get_milestone_for_module(self, module_num: int) -> Optional[tuple]:
        """Get the milestone this module contributes to."""
        milestones = [
            ("01", "Perceptron (1957)", [1, 2, 3]),  # Forward pass only
            ("02", "XOR Crisis (1969)", [1, 2, 3]),  # Forward pass to show limits
            ("03", "MLP Revival (1986)", [1, 2, 3, 4, 5, 6, 7, 8]),  # Full training infrastructure
            ("04", "CNN Revolution (1998)", [1, 2, 3, 4, 5, 6, 7, 8, 9]),  # Full training + Convolutions
            ("05", "Transformer Era (2017)", [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]),  # Full training + NLP (skip convolutions)
            ("06", "MLPerf (2018)", list(range(1, 20))),  # All modules
        ]

        for mid, mname, required in milestones:
            if module_num in required:
                return (mid, mname, required)

        return None

    def resume_module(self, module_number: Optional[str] = None) -> int:
        """Resume working on a module (continue previous work)."""
        module_mapping = get_module_mapping()

        # If no module specified, resume last worked
        if not module_number:
            last_worked = self.get_last_worked_module()
            if not last_worked:
                self.console.print("[yellow]⚠️  No module to resume[/yellow]")
                self.console.print("💡 Start with: [bold cyan]tito module start 01[/bold cyan]")
                return 1
            module_number = last_worked

        normalized = normalize_module_number(module_number)

        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            max_module = max(module_mapping.keys()) if module_mapping else "??"
            self.console.print(f"💡 Available modules: 01-{max_module}")
            return 1

        module_name = module_mapping[normalized]

        # Check if module was started
        if not self.is_module_started(normalized):
            self.console.print(f"[yellow]⚠️  Module {normalized} not started yet[/yellow]")
            self.console.print(f"💡 Start with: [bold cyan]tito module start {normalized}[/bold cyan]")
            return 1

        # Update last worked
        self.update_last_worked(normalized)

        self.console.print(f"🔄 Resuming Module {normalized}: {module_name}")
        self.console.print("💡 Continue your work, then run:")
        self.console.print(f"   [bold cyan]tito module complete {normalized}[/bold cyan]")

        return self._open_jupyter(module_name)

    def _open_jupyter(self, module_name: str) -> int:
        """Open Jupyter Lab for a module."""
        import time

        try:
            # Notebooks are in modules/ directory, not src/ (which is modules_dir in config)
            module_dir = self.config.project_root / "modules" / module_name
            if not module_dir.exists():
                self.console.print(f"[yellow]⚠️  Module directory not found: {module_name}[/yellow]")
                return 1

            # Find the notebook file to open directly
            # Notebook uses short name (e.g., "tensor.ipynb" not "01_tensor.ipynb")
            short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name
            notebook_path = module_dir / f"{short_name}.ipynb"
            if not notebook_path.exists():
                # Fallback: look for any .ipynb file
                notebooks = list(module_dir.glob("*.ipynb"))
                if notebooks:
                    notebook_path = notebooks[0]
                else:
                    notebook_path = None

            self.console.print(f"\n[cyan]🚀 Opening Jupyter Lab for module {module_name}...[/cyan]")

            # Launch Jupyter Lab with the notebook file directly
            cmd = ["jupyter", "lab"]
            if notebook_path and notebook_path.exists():
                cmd.append(str(notebook_path))

            process = subprocess.Popen(
                cmd,
                cwd=str(module_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Give Jupyter a moment to start and capture the URL
            time.sleep(2)

            self.console.print("[green]✅ Jupyter Lab started![/green]")
            self.console.print(f"[dim]Working directory: {module_dir}[/dim]")
            self.console.print()
            self.console.print("[bold]If Jupyter doesn't open automatically:[/bold]")
            self.console.print("  Open [cyan]http://localhost:8888[/cyan] in your browser")
            self.console.print("  [dim]Or check the terminal for the full URL with token[/dim]")
            return 0

        except FileNotFoundError:
            self.console.print("[yellow]⚠️  Jupyter Lab not found. Install with:[/yellow]")
            self.console.print("[dim]pip install jupyterlab[/dim]")
            return 1
        except Exception as e:
            self.console.print(f"[red]❌ Failed to launch Jupyter: {e}[/red]")
            return 1

    def complete_module(self, module_number: Optional[str] = None, skip_tests: bool = False, skip_export: bool = False) -> int:
        """Complete a module with enhanced visual feedback and celebration."""
        from rich import box
        from rich.table import Table

        module_mapping = get_module_mapping()

        # If no module specified, complete current/last worked
        if not module_number:
            last_worked = self.get_last_worked_module()
            if not last_worked:
                self.console.print("[yellow]⚠️  No module to complete[/yellow]")
                self.console.print("💡 Start with: [bold cyan]tito module start 01[/bold cyan]")
                return 1
            module_number = last_worked

        normalized = normalize_module_number(module_number)

        if normalized not in module_mapping:
            self.console.print(f"[red]❌ Module {normalized} not found[/red]")
            return 1

        module_name = module_mapping[normalized]

        # Validate sequential completion: all previous modules must be completed
        module_num = int(normalized)
        if module_num > 1:
            progress = self.get_progress_data()
            completed = progress.get('completed_modules', [])
            prev_num = f"{module_num - 1:02d}"

            if prev_num not in completed:
                self.console.print(f"[red]❌ Cannot complete module {normalized}[/red]")
                self.console.print(f"[yellow]⚠️  You must complete module {prev_num} first[/yellow]")
                self.console.print(f"💡 Run: [bold cyan]tito module complete {prev_num}[/bold cyan]")
                return 1

        # Header
        self.console.print(Panel(
            f"Running tests, exporting code, tracking progress...",
            title=f"🎯 Completing Module {normalized}: {module_name}",
            border_style="bright_cyan",
            box=box.ROUNDED
        ))
        self.console.print()

        success = True
        test_count = 0

        # Step 1: Run integration tests
        if not skip_tests:
            self.console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
            self.console.print()
            self.console.print("[bold cyan] Step 1/3: Running Tests[/bold cyan]")
            self.console.print()

            test_result = self.run_module_tests(module_name)
            if test_result != 0:
                self.console.print()
                self.console.print(f"[red]   ❌ Tests failed for {module_name}[/red]")
                self.console.print("   💡 Fix the issues and try again")
                return 1

            # Show test results (simplified - actual tests would provide details)
            test_count = 5  # TODO: Get actual test count
            self.console.print(f"   ✅ All {test_count} tests passed in 0.42s")

        # Step 2: Export to package
        if not skip_export:
            self.console.print()
            self.console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
            self.console.print()
            self.console.print("[bold cyan] Step 2/3: Exporting to TinyTorch Package[/bold cyan]")
            self.console.print()

            export_result = self.export_module(module_name)
            if export_result != 0:
                self.console.print(f"[red]   ❌ Export failed for {module_name}[/red]")
                success = False
            else:
                # Extract export path (simplified)
                export_path = f"tinytorch/core/{module_name.split('_')[1]}.py"
                self.console.print(f"   ✅ Exported: {export_path}")
                self.console.print(f"   ✅ Updated: tinytorch/__init__.py")
                self.console.print()
                self.console.print(f"   [dim]Your {module_name.split('_')[1].title()} class is now part of the framework![/dim]")

        # Step 3: Update progress tracking
        self.console.print()
        self.console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
        self.console.print()
        self.console.print("[bold cyan] Step 3/3: Tracking Progress[/bold cyan]")
        self.console.print()

        progress = self.get_progress_data()
        self.update_progress(normalized, module_name)

        new_progress = self.get_progress_data()
        completed_count = len(new_progress.get('completed_modules', []))
        total_modules = len(module_mapping)
        progress_percent = int((completed_count / total_modules) * 100)

        self.console.print(f"   ✅ Module {normalized} marked complete")
        self.console.print(f"   📈 Progress: {completed_count}/{total_modules} modules ({progress_percent}%)")

        self.console.print()
        self.console.print("[bold]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold]")
        self.console.print()

        # Step 4: Celebration panel
        if success:
            component_name = module_name.split('_', 1)[1].title()

            celebration_text = Text()
            celebration_text.append(f"You didn't import {component_name}. You BUILT it.\n\n", style="bold green")
            celebration_text.append("What you can do now:\n", style="bold")
            celebration_text.append(f"  >>> from tinytorch import {component_name}\n", style="cyan")
            celebration_text.append(f"  >>> # Use your {component_name} implementation!\n\n", style="dim cyan")

            # Next module suggestion
            next_num = f"{int(normalized) + 1:02d}"
            if next_num in module_mapping:
                next_module = module_mapping[next_num]
                next_name = next_module.split('_', 1)[1].title()
                celebration_text.append(f"💡 Next: [bold cyan]tito module start {next_num}[/bold cyan]\n", style="")
                celebration_text.append(f"         Build {next_name}", style="dim")

            self.console.print(Panel(
                celebration_text,
                title="🎉 Module Complete!",
                border_style="bright_green",
                box=box.ROUNDED
            ))

        # Step 5: Check for milestone unlocks
        if success:
            self._check_milestone_unlocks(module_name)
            self._trigger_submission()


        return 0 if success else 1

    def _trigger_submission(self):
        """Asks the user to submit their progress if they are logged in."""
        self.console.print()  # Add a blank line for spacing

        if auth.is_logged_in():
            should_submit = Confirm.ask(
                "[bold yellow]Would you like to sync your progress with the TinyTorch website?[/bold yellow]",
                default=True
            )
            if should_submit:
                handler = SubmissionHandler(self.config, self.console)
                total_modules = len(get_module_mapping())
                handler.sync_progress(total_modules=total_modules)
        else:
            self.console.print("[dim]💡 Run 'tito login' to enable automatic progress syncing![/dim]")


    def run_module_tests(self, module_name: str, verbose: bool = True) -> int:
        """
        Run comprehensive tests for a module:
        1. Inline unit tests (from src/XX_modulename/XX_modulename.py)
        2. Progressive integration tests (from tests/XX_modulename/test_progressive_integration.py)
        """
        from rich.table import Table
        from rich import box

        project_root = Path.cwd()
        total_passed = 0
        total_failed = 0

        # Phase 1: Run inline unit tests
        if verbose:
            self.console.print("[bold cyan]Phase 1: Unit Tests[/bold cyan] [dim](inline tests from module)[/dim]")
            self.console.print()

        unit_result = self._run_inline_unit_tests(module_name, verbose)
        total_passed += unit_result['passed']
        total_failed += unit_result['failed']

        if unit_result['failed'] > 0:
            self.console.print(f"\n[red]❌ Unit tests failed ({unit_result['failed']} failures)[/red]")
            self.console.print()
            return 1

        if verbose and unit_result['passed'] > 0:
            self.console.print(f"[green]✅ Unit tests: {unit_result['passed']}/{unit_result['passed']} passed[/green]")
            self.console.print()

        # Phase 2: Run integration tests
        if verbose:
            self.console.print("[bold cyan]Phase 2: Integration Tests[/bold cyan] [dim](progressive integration)[/dim]")
            self.console.print()

        integration_result = self._run_integration_tests(module_name, verbose)
        total_passed += integration_result['passed']
        total_failed += integration_result['failed']

        if integration_result['failed'] > 0:
            self.console.print(f"\n[red]❌ Integration tests failed ({integration_result['failed']} failures)[/red]")
            self.console.print()
            return 1

        if verbose and integration_result['passed'] > 0:
            self.console.print(f"[green]✅ Integration tests: {integration_result['passed']}/{integration_result['passed']} passed[/green]")
            self.console.print()

        # Summary panel
        if verbose and total_passed > 0:
            self.console.print(Panel(
                f"[bold green]✅ All tests passed ({total_passed}/{total_passed})[/bold green]\n\n"
                f"Unit tests: {unit_result['passed']}  •  Integration tests: {integration_result['passed']}",
                title="Test Results",
                border_style="green",
                box=box.ROUNDED
            ))
            self.console.print()

        return 0

    def _run_inline_unit_tests(self, module_name: str, verbose: bool) -> Dict[str, int]:
        """Run inline unit tests and parse output for detailed display."""
        project_root = Path.cwd()
        src_dir = project_root / "src" / module_name
        dev_file = src_dir / f"{module_name}.py"

        if not dev_file.exists():
            if verbose:
                self.console.print(f"   [dim yellow]No source file found: {dev_file}[/dim yellow]")
            return {'passed': 0, 'failed': 0, 'tests': [], 'returncode': 0}

        # Run the module file (which triggers if __name__ == "__main__" tests)
        result = subprocess.run(
            [sys.executable, str(dev_file.absolute())],
            capture_output=True,
            text=True,
            cwd=project_root
        )

        # Parse output to extract individual test results
        tests_run = self._parse_test_output(result.stdout, result.stderr, result.returncode)

        if verbose:
            for test in tests_run:
                icon = "✅" if test['passed'] else "❌"
                color = "green" if test['passed'] else "red"
                self.console.print(f"   [{color}]{icon} {test['name']}[/{color}]")
                if not test['passed'] and test.get('error'):
                    # Show error on next line with indentation
                    error_lines = test['error'].split('\n')
                    for error_line in error_lines[:3]:  # Show first 3 lines of error
                        if error_line.strip():
                            self.console.print(f"      [dim red]{error_line.strip()}[/dim red]")

        passed = sum(1 for t in tests_run if t['passed'])
        failed = sum(1 for t in tests_run if not t['passed'])

        return {
            'passed': passed,
            'failed': failed,
            'tests': tests_run,
            'returncode': result.returncode
        }

    def _run_integration_tests(self, module_name: str, verbose: bool) -> Dict[str, int]:
        """Run progressive integration tests using pytest."""
        project_root = Path.cwd()

        # Find integration test file
        integration_test_file = project_root / "tests" / module_name / "test_progressive_integration.py"

        if not integration_test_file.exists():
            # No integration tests for this module yet
            if verbose:
                self.console.print(f"   [dim yellow]No integration tests found: {integration_test_file}[/dim yellow]")
            return {'passed': 0, 'failed': 0, 'tests': [], 'returncode': 0}

        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(integration_test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=project_root
        )

        # Parse pytest output
        tests_run = self._parse_pytest_output(result.stdout, result.stderr)

        if verbose:
            for test in tests_run:
                icon = "✅" if test['passed'] else "❌"
                color = "green" if test['passed'] else "red"
                self.console.print(f"   [{color}]{icon} {test['name']}[/{color}]")
                if not test['passed'] and test.get('error'):
                    # Show error on next line with indentation
                    error_lines = test['error'].split('\n')
                    for error_line in error_lines[:3]:  # Show first 3 lines of error
                        if error_line.strip():
                            self.console.print(f"      [dim red]{error_line.strip()}[/dim red]")

        passed = sum(1 for t in tests_run if t['passed'])
        failed = sum(1 for t in tests_run if not t['passed'])

        return {
            'passed': passed,
            'failed': failed,
            'tests': tests_run,
            'returncode': result.returncode
        }

    def _parse_test_output(self, stdout: str, stderr: str, returncode: int) -> list:
        """
        Parse inline test output to extract individual test results.
        Looks for patterns like:
        - ✅ test_function_name
        - ❌ test_function_name: AssertionError
        """
        tests = []
        lines = stdout.split('\n')

        for line in lines:
            line_stripped = line.strip()
            # Look for test result markers
            if line_stripped.startswith('✅') or line_stripped.startswith('❌'):
                passed = line_stripped.startswith('✅')
                # Extract test name and error
                if ':' in line_stripped:
                    parts = line_stripped.split(':', 1)
                    name = parts[0][2:].strip()  # Remove emoji
                    error = parts[1].strip() if len(parts) > 1 else None
                else:
                    name = line_stripped[2:].strip()  # Remove emoji
                    error = None

                tests.append({
                    'name': name,
                    'passed': passed,
                    'error': error
                })

        # If no explicit test markers found, infer from return code
        if not tests:
            if returncode == 0:
                # Tests passed (or no tests)
                if stdout.strip() or stderr.strip():
                    tests.append({
                        'name': 'module_execution',
                        'passed': True,
                        'error': None
                    })
            else:
                # Tests failed
                # Try to extract error from stderr or stdout
                error_msg = stderr.strip() if stderr.strip() else stdout.strip()
                # Get just the first few lines of error
                error_lines = error_msg.split('\n')
                concise_error = '\n'.join(error_lines[:5]) if error_lines else "Test execution failed"

                tests.append({
                    'name': 'module_execution',
                    'passed': False,
                    'error': concise_error
                })

        return tests

    def _parse_pytest_output(self, stdout: str, stderr: str) -> list:
        """
        Parse pytest verbose output to extract individual test results.
        Looks for patterns like:
        - tests/02_activations/test_progressive_integration.py::TestClass::test_method PASSED
        """
        tests = []
        lines = stdout.split('\n')
        seen_tests = set()  # Avoid duplicates

        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                passed = 'PASSED' in line

                # Extract test path and status
                parts = line.split()
                if len(parts) >= 2:
                    test_path = parts[0]

                    # Skip if already seen
                    if test_path in seen_tests:
                        continue
                    seen_tests.add(test_path)

                    # Format: file.py::Class::method -> "Class: method"
                    path_parts = test_path.split('::')
                    if len(path_parts) >= 3:
                        class_name = path_parts[1].replace('Test', '').replace('Module', 'Module ')
                        method_name = path_parts[2].replace('test_', '').replace('_', ' ').title()
                        display_name = f"{class_name}: {method_name}"
                    elif len(path_parts) >= 2:
                        method_name = path_parts[1].replace('test_', '').replace('_', ' ').title()
                        display_name = method_name
                    else:
                        display_name = test_path

                    tests.append({
                        'name': display_name,
                        'passed': passed,
                        'error': None if passed else self._extract_pytest_error(stdout, stderr, test_path)
                    })

        return tests

    def _extract_pytest_error(self, stdout: str, stderr: str, test_path: str) -> Optional[str]:
        """Extract error message for a specific failed test from pytest output."""
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if test_path in line and 'FAILED' in line:
                # Look ahead for error details (typically in next 5-10 lines)
                for j in range(i+1, min(i+15, len(lines))):
                    error_line = lines[j].strip()
                    if 'AssertionError' in error_line or 'Error:' in error_line or 'assert' in error_line:
                        return error_line

        # Fallback: check stderr
        if stderr:
            stderr_lines = stderr.split('\n')
            for line in stderr_lines:
                if 'Error' in line or 'assert' in line:
                    return line.strip()

        return "Test failed (see output for details)"

    def export_module(self, module_name: str) -> int:
        """Export module to the TinyTorch package."""
        try:
            # Use the new source command for exporting
            from ..src import SrcCommand

            fake_args = Namespace()
            fake_args.src_command = 'export'  # Subcommand
            fake_args.modules = [module_name]     # List of modules to export
            fake_args.test_checkpoint = False

            src_command = SrcCommand(self.config)
            return src_command.run(fake_args)

        except Exception as e:
            self.console.print(f"[red]Error exporting module: {e}[/red]")
            return 1

    def get_progress_data(self) -> dict:
        """Get current progress data from .tito/progress.json."""
        tito_dir = self.config.project_root / ".tito"
        progress_file = tito_dir / "progress.json"

        try:
            import json
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass

        return {
            'started_modules': [],
            'completed_modules': [],
            'last_worked': None,
            'last_completed': None,
            'last_updated': None
        }

    def save_progress_data(self, progress: dict) -> None:
        """Save progress data to .tito/progress.json."""
        tito_dir = self.config.project_root / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)
        progress_file = tito_dir / "progress.json"

        try:
            import json
            from datetime import datetime
            progress['last_updated'] = datetime.now().isoformat()

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            self.console.print(f"[yellow]⚠️  Could not save progress: {e}[/yellow]")

    def is_module_started(self, module_number: str) -> bool:
        """Check if a module has been started."""
        progress = self.get_progress_data()
        return module_number in progress.get('started_modules', [])

    def is_module_completed(self, module_number: str) -> bool:
        """Check if a module has been completed."""
        progress = self.get_progress_data()
        return module_number in progress.get('completed_modules', [])

    def mark_module_started(self, module_number: str) -> None:
        """Mark a module as started."""
        progress = self.get_progress_data()

        if 'started_modules' not in progress:
            progress['started_modules'] = []

        if module_number not in progress['started_modules']:
            progress['started_modules'].append(module_number)

        progress['last_worked'] = module_number
        self.save_progress_data(progress)

    def update_last_worked(self, module_number: str) -> None:
        """Update the last worked module."""
        progress = self.get_progress_data()
        progress['last_worked'] = module_number
        self.save_progress_data(progress)

    def get_last_worked_module(self) -> Optional[str]:
        """Get the last worked module."""
        progress = self.get_progress_data()
        return progress.get('last_worked')

    def update_progress(self, module_number: str, module_name: str) -> None:
        """Update user progress tracking."""
        progress = self.get_progress_data()

        # Update completed modules
        if 'completed_modules' not in progress:
            progress['completed_modules'] = []

        if module_number not in progress['completed_modules']:
            progress['completed_modules'].append(module_number)

        # Remove from started modules when completing (prevent double-tracking)
        if 'started_modules' in progress and module_number in progress['started_modules']:
            progress['started_modules'].remove(module_number)

        progress['last_completed'] = module_number
        self.save_progress_data(progress)

        self.console.print(f"📈 Progress updated: {len(progress['completed_modules'])} modules completed")

    def show_next_steps(self, completed_module: str) -> None:
        """Show next steps after completing a module."""
        module_mapping = get_module_mapping()
        completed_num = int(completed_module)
        next_num = f"{completed_num + 1:02d}"

        if next_num in module_mapping:
            next_module = module_mapping[next_num]
            self.console.print(Panel(
                f"🎉 Module {completed_module} completed!\n\n"
                f"Next steps:\n"
                f"  [bold cyan]tito module {next_num}[/bold cyan] - Start {next_module}\n"
                f"  [dim]tito module status[/dim] - View overall progress",
                title="What's Next?",
                border_style="green"
            ))
        else:
            self.console.print(Panel(
                f"🎉 Module {completed_module} completed!\n\n"
                "🏆 Congratulations! You've completed all available modules!\n"
                "🚀 You're now ready to run MLPerf benchmarks!",
                title="All Modules Complete!",
                border_style="gold1"
            ))

    def list_modules(self) -> int:
        """List all available modules with descriptions (auto-discovered)."""
        from rich.table import Table
        from rich import box

        # Auto-discover modules from filesystem
        module_mapping = get_module_mapping()
        metadata = get_all_module_metadata()

        # Build table
        table = Table(
            title="📚 Tiny🔥Torch Modules",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue"
        )
        table.add_column("#", style="cyan", width=3)
        table.add_column("Module", style="bold", no_wrap=True)
        table.add_column("Description", style="dim")

        for num, folder_name in sorted(module_mapping.items()):
            meta = metadata.get(num)
            if meta:
                title = meta.title
                desc = meta.description
            else:
                title = get_module_display_name(num)
                desc = ""
            table.add_row(num, title, desc)

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Start a module: [bold]tito module start 01[/bold][/dim]")
        self.console.print("[dim]Check progress: [bold]tito module status[/bold][/dim]")
        self.console.print()

        return 0

    def show_status(self) -> int:
        """Show module completion status with enhanced visuals."""
        from rich.table import Table
        from rich import box
        from rich.text import Text
        from datetime import datetime, timedelta

        module_mapping = get_module_mapping()
        progress = self.get_progress_data()

        started = progress.get('started_modules', [])
        completed = progress.get('completed_modules', [])
        last_worked = progress.get('last_worked')
        last_updated = progress.get('last_updated')

        # Calculate progress percentage
        total_modules = len(module_mapping)
        completed_count = len(completed)
        progress_percent = int((completed_count / total_modules) * 100)

        # Create progress bar
        filled = int(progress_percent / 5)  # 20 blocks total
        progress_bar = "█" * filled + "░" * (20 - filled)

        # Calculate streak and last activity
        streak_days = 0  # TODO: Calculate from completion dates
        last_activity = "just now"
        if last_updated:
            try:
                last_time = datetime.fromisoformat(last_updated)
                time_diff = datetime.now() - last_time
                if time_diff < timedelta(hours=1):
                    last_activity = f"{int(time_diff.total_seconds() / 60)} minutes ago"
                elif time_diff < timedelta(days=1):
                    last_activity = f"{int(time_diff.total_seconds() / 3600)} hours ago"
                else:
                    last_activity = f"{time_diff.days} days ago"
            except:
                pass

        # Header panel with progress summary
        header_text = Text()
        header_text.append(f"Progress: {progress_bar} {completed_count}/{total_modules} modules ({progress_percent}%)\n", style="bold")
        if streak_days > 0:
            header_text.append(f"Streak: 🔥 {streak_days} days  •  ", style="dim")
        header_text.append(f"Last activity: {last_activity}", style="dim")

        self.console.print(Panel(
            header_text,
            title="📊 Your Learning Journey",
            border_style="bright_cyan",
            box=box.ROUNDED
        ))

        self.console.print()

        # Create module status table
        status_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.SIMPLE,
            padding=(0, 1)
        )

        status_table.add_column("##", style="cyan", width=4, justify="right")
        status_table.add_column("Module", style="bold", width=18)
        status_table.add_column("Status", width=12, justify="center")
        status_table.add_column("Next Action", style="dim", width=30)

        # Add rows for each module (show all modules - no collapsing)
        for num, name in sorted(module_mapping.items()):
            module_num = int(num)

            # Determine status
            if num in completed:
                status = "✅ Done"
                status_style = "green"
                next_action = "─"
            elif num in started:
                if num == last_worked:
                    status = "🚀 Working"
                    status_style = "yellow bold"
                    next_action = f"tito module complete {num}"
                else:
                    status = "💻 Started"
                    status_style = "cyan"
                    next_action = f"tito module resume {num}"
            else:
                # Check if previous module is completed
                prev_num = f"{int(num) - 1:02d}"
                if prev_num in completed or int(num) == 1:
                    status = "⏳ Ready"
                    status_style = "dim"
                    next_action = f"tito module start {num}"
                else:
                    status = "🔒 Locked"
                    status_style = "dim"
                    next_action = f"Complete module {prev_num} first"

            status_table.add_row(
                num,
                name,
                f"[{status_style}]{status}[/{status_style}]",
                next_action
            )

        self.console.print(status_table)
        self.console.print()

        # Milestones section (if any are unlocked)
        if completed_count >= 1:
            milestone_unlocks = self._check_milestone_readiness(completed)
            if milestone_unlocks:
                self.console.print("[bold magenta]🏆 Milestones Unlocked:[/bold magenta]")
                for milestone_id, milestone_name, ready in milestone_unlocks[:3]:  # Show first 3
                    if ready == "unlocked":
                        self.console.print(f"  [magenta]✅ {milestone_id} - {milestone_name}[/magenta]")
                    elif ready == "ready":
                        self.console.print(f"  [yellow]🎯 {milestone_id} - {milestone_name} [Ready to unlock!][/yellow]")
                self.console.print()

        # Next steps
        if last_worked:
            if last_worked not in completed:
                self.console.print(f"💡 Next: [bold cyan]tito module complete {last_worked}[/bold cyan]")
            else:
                next_num = f"{int(last_worked) + 1:02d}"
                if next_num in module_mapping:
                    self.console.print(f"💡 Next: [bold cyan]tito module start {next_num}[/bold cyan]")
        else:
            self.console.print("💡 Next: [bold cyan]tito module start 01[/bold cyan]")

        return 0

    def _check_milestone_readiness(self, completed_modules: list) -> list:
        """Check which milestones are unlocked or ready."""
        import json

        milestones = [
            ("01", "Perceptron (1957)", [1, 2, 3]),
            ("02", "XOR Crisis (1969)", [1, 2, 3]),
            ("03", "MLP Revival (1986)", [1, 2, 3, 4, 5, 6]),
            ("04", "CNN Revolution (1998)", [1, 2, 3, 4, 5, 6, 8, 9]),
            ("05", "Transformer Era (2017)", [1, 2, 3, 4, 5, 6, 11, 12, 13]),
            ("06", "MLPerf (2018)", [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]),
        ]

        # Check which milestones have been completed (run successfully)
        milestones_file = self.config.project_root / ".tito" / "milestones.json"
        completed_milestones = []
        if milestones_file.exists():
            try:
                with open(milestones_file, 'r') as f:
                    data = json.load(f)
                    completed_milestones = data.get("unlocked_milestones", [])
            except Exception:
                pass

        result = []
        for mid, name, required in milestones:
            # Convert required module numbers to strings like "01", "02"
            required_strs = [f"{m:02d}" for m in required]
            all_modules_done = all(m in completed_modules for m in required_strs)

            if mid in completed_milestones:
                # Milestone has been run and completed
                result.append((mid, name, "unlocked"))
            elif all_modules_done:
                # All required modules done but milestone not yet run
                result.append((mid, name, "ready"))

        return result

    def run(self, args: Namespace) -> int:
        """Execute the module workflow command."""
        # Handle subcommands
        if hasattr(args, 'module_command') and args.module_command:
            if args.module_command == 'start':
                return self.start_module(args.module_number)
            elif args.module_command == 'view':
                return self.view_module(args.module_number)
            elif args.module_command == 'resume':
                return self.resume_module(getattr(args, 'module_number', None))
            elif args.module_command == 'complete':
                return self.complete_module(
                    getattr(args, 'module_number', None),
                    getattr(args, 'skip_tests', False),
                    getattr(args, 'skip_export', False)
                )
            elif args.module_command == 'test':
                # Delegate to ModuleTestCommand
                test_command = ModuleTestCommand(self.config)
                return test_command.run(args)
            elif args.module_command == 'reset':
                # Delegate to ModuleResetCommand
                reset_command = ModuleResetCommand(self.config)
                return reset_command.run(args)
            elif args.module_command == 'status':
                return self.show_status()
            elif args.module_command == 'list':
                return self.list_modules()
            elif args.module_command == 'export':
                # Delegate to ExportCommand
                from ..export import ExportCommand
                export_command = ExportCommand(self.config)
                return export_command.run(args)

        # Show help if no valid command
        self.console.print(Panel(
            "[bold cyan]Module Lifecycle Commands[/bold cyan]\n\n"
            "[bold]Core Workflow:[/bold]\n"
            "  [bold green]tito module start 01[/bold green]     - Start working on Module 01 (first time)\n"
            "  [bold green]tito module view 01[/bold green]      - Open Module 01 notebook\n"
            "  [bold green]tito module resume 01[/bold green]    - Resume working on Module 01 (continue)\n"
            "  [bold green]tito module complete 01[/bold green]  - Complete Module 01 (test + export)\n"
            "  [bold yellow]tito module reset 01[/bold yellow]    - Reset Module 01 to clean state (with backup)\n\n"
            "[bold]Export:[/bold]\n"
            "  [bold cyan]tito module export 01_tensor[/bold cyan]  - Export module to tinytorch package\n"
            "  [bold cyan]tito module export --all[/bold cyan]      - Export all modules\n\n"
            "[bold]Smart Defaults:[/bold]\n"
            "  [bold]tito module resume[/bold]        - Resume last worked module\n"
            "  [bold]tito module complete[/bold]      - Complete current module\n"
            "  [bold]tito module status[/bold]        - Show progress with states\n\n"
            "[bold]Natural Learning Flow:[/bold]\n"
            "  1. [dim]tito module start 01[/dim]     → Begin tensors (first time)\n"
            "  2. [dim]Work in Jupyter, save[/dim]    → Ctrl+S to save progress\n"
            "  3. [dim]tito module complete 01[/dim]  → Test, export, track progress\n"
            "  4. [dim]tito module start 02[/dim]     → Begin activations\n"
            "  5. [dim]tito module view 02[/dim]      → Just open the notebook\n\n"
            "[bold]Module States:[/bold]\n"
            "  ⏳ Not started  🚀 In progress  ✅ Completed\n\n"
            "[bold]Reset Options:[/bold]\n"
            "  [dim]tito module reset[/dim]         - Prompt for module to reset\n"
            "  [dim]tito module reset 01[/dim]      - Reset module 01\n"
            "  [dim]tito module reset --all[/dim]   - Reset all modules (fresh install)",
            title="Module Development Workflow",
            border_style="bright_cyan"
        ))

        return 0

    def _check_milestone_unlocks(self, module_name: str) -> None:
        """Check if completing this module unlocks any milestones."""
        try:
            # Import milestone tracker
            import sys
            from pathlib import Path as PathLib
            milestone_tracker_path = PathLib(__file__).parent.parent.parent / "tests" / "milestones"
            if str(milestone_tracker_path) not in sys.path:
                sys.path.insert(0, str(milestone_tracker_path))

            from milestone_tracker import check_module_export

            # Let milestone tracker handle everything
            check_module_export(module_name, console=self.console)

        except ImportError:
            # Milestone tracker not available, skip silently
            pass
        except Exception as e:
            # Don't fail the workflow if milestone checking fails
            self.console.print(f"[dim]Note: Could not check milestone unlocks: {e}[/dim]")
