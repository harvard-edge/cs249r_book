"""
Module Reset Command for TinyTorch CLI.

Simple reset functionality:
- Reset a specific module to pristine state (recreate notebook from src/)
- Reset all modules (fresh install)
"""

import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from ..base import BaseCommand
from ..export_utils import convert_py_to_notebook
from ...core.modules import get_module_mapping, normalize_module_number


class ModuleResetCommand(BaseCommand):
    """Command to reset a module to clean state."""

    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset module to clean state"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add reset command arguments."""
        parser.add_argument(
            "module_number", nargs="?", default=None,
            help="Module number to reset (01, 02, etc.)"
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Reset ALL modules to pristine state",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Skip confirmation prompts"
        )

    def _prompt_for_module(self) -> Optional[str]:
        """Prompt user to select a module to reset."""
        console = self.console
        module_mapping = get_module_mapping()

        console.print()
        console.print("[bold cyan]Which module do you want to reset?[/bold cyan]")
        console.print()

        # Show available modules
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Module", style="bold")

        for num, name in sorted(module_mapping.items()):
            table.add_row(num, name)

        console.print(table)
        console.print()

        try:
            response = input("Enter module number (or 'q' to cancel): ").strip()
            if response.lower() in ['q', 'quit', 'cancel', '']:
                return None
            return response
        except (KeyboardInterrupt, EOFError):
            console.print()
            return None

    def _confirm_reset(self, message: str) -> bool:
        """Ask user for confirmation."""
        console = self.console
        console.print()
        console.print(f"[yellow]{message}[/yellow]")
        console.print()

        try:
            response = input("Are you sure? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except (KeyboardInterrupt, EOFError):
            console.print()
            return False

    def _reset_single_module(self, module_number: str) -> int:
        """Reset a single module by recreating notebook from src/."""
        console = self.console
        module_mapping = get_module_mapping()

        normalized = normalize_module_number(module_number)
        if normalized not in module_mapping:
            console.print(f"[red]Invalid module number: {module_number}[/red]")
            console.print(f"Available modules: 01-{max(module_mapping.keys())}")
            return 1

        module_name = module_mapping[normalized]

        # Check if source exists
        src_path = self.config.project_root / "src" / module_name
        if not src_path.exists():
            console.print(f"[red]Source not found: src/{module_name}[/red]")
            return 1

        console.print()
        console.print(f"[cyan]Resetting module {normalized}: {module_name}[/cyan]")
        console.print()

        # Convert src/ to notebook in modules/
        success = convert_py_to_notebook(src_path, self.venv_path, console)

        if success:
            # Update progress tracking (remove from completed)
            self._update_progress_tracking(normalized, module_name)

            console.print()
            console.print(
                Panel(
                    f"[bold green]✅ Module {normalized} reset successfully![/bold green]\n\n"
                    f"The notebook has been recreated from source.\n\n"
                    f"[bold]Next steps:[/bold]\n"
                    f"  • [dim]tito module start {normalized}[/dim] - Begin working\n"
                    f"  • [dim]tito module view {normalized}[/dim] - Open the notebook",
                    title="Reset Complete",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(f"[red]Failed to reset module {module_name}[/red]")
            return 1

    def _reset_all_modules(self) -> int:
        """Reset all modules to pristine state."""
        console = self.console
        module_mapping = get_module_mapping()

        console.print()
        console.print(
            Panel(
                f"[bold red]⚠️  RESET ALL MODULES[/bold red]\n\n"
                f"This will:\n"
                f"  • Reset ALL {len(module_mapping)} modules to pristine state\n"
                f"  • Clear all progress tracking\n"
                f"  • Recreate all notebooks from source\n\n"
                f"[dim]This is like a fresh install.[/dim]",
                title="Warning",
                border_style="red",
            )
        )

        reset_count = 0
        failed_modules = []

        for module_num, module_name in sorted(module_mapping.items()):
            console.print(f"[cyan]Resetting {module_name}...[/cyan]")

            src_path = self.config.project_root / "src" / module_name
            if not src_path.exists():
                console.print(f"[yellow]  ⚠ Source not found, skipping[/yellow]")
                continue

            success = convert_py_to_notebook(src_path, self.venv_path, console)
            if success:
                console.print(f"[green]  ✓ {module_name} reset[/green]")
                reset_count += 1
            else:
                console.print(f"[red]  ✗ {module_name} failed[/red]")
                failed_modules.append(module_name)

        # Reset progress tracking
        console.print()
        console.print("[cyan]Clearing progress tracking...[/cyan]")
        self._clear_all_progress()
        console.print("[green]  ✓ Progress cleared[/green]")

        # Summary
        console.print()
        if failed_modules:
            console.print(
                Panel(
                    f"[yellow]⚠ Partial reset complete[/yellow]\n\n"
                    f"Reset: {reset_count} modules\n"
                    f"Failed: {len(failed_modules)} modules\n\n"
                    f"[dim]Failed: {', '.join(failed_modules)}[/dim]",
                    title="Reset Complete (with errors)",
                    border_style="yellow",
                )
            )
            return 1
        else:
            console.print(
                Panel(
                    f"[bold green]✅ All {reset_count} modules reset![/bold green]\n\n"
                    f"You now have a fresh TinyTorch install.\n\n"
                    f"[bold]Next steps:[/bold]\n"
                    f"  • [dim]tito module start 01[/dim] - Begin your journey\n"
                    f"  • [dim]tito module status[/dim] - Check status",
                    title="Fresh Install State",
                    border_style="green",
                )
            )
            return 0

    def _update_progress_tracking(self, module_number: str, module_name: str) -> None:
        """Remove module from completed progress in .tito/progress.json."""
        console = self.console

        tito_dir = self.config.project_root / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)
        progress_file = tito_dir / "progress.json"

        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    progress = json.load(f)

                # Remove from completed and started modules
                if "completed_modules" in progress:
                    if module_number in progress["completed_modules"]:
                        progress["completed_modules"].remove(module_number)

                if "started_modules" in progress:
                    if module_number in progress["started_modules"]:
                        progress["started_modules"].remove(module_number)

                progress["last_updated"] = datetime.now().isoformat()

                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)
            except Exception as e:
                console.print(f"[dim]Could not update progress: {e}[/dim]")

    def _clear_all_progress(self) -> None:
        """Clear all progress tracking in .tito/ directory."""
        tito_dir = self.config.project_root / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)

        # Reset .tito/progress.json
        progress_file = tito_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "version": "1.0",
            "started_modules": [],
            "completed_modules": [],
            "last_worked": None,
            "last_updated": datetime.now().isoformat()
        }, indent=2))

        # Reset milestones
        milestones_file = tito_dir / "milestones.json"
        milestones_file.write_text(json.dumps({
            "version": "1.0",
            "completed_milestones": [],
            "completion_dates": {}
        }, indent=2))

    def run(self, args: Namespace) -> int:
        """Execute the reset command."""
        console = self.console

        # Handle --all (reset all modules)
        if getattr(args, 'all', False):
            if not args.force:
                if not self._confirm_reset("This will reset ALL modules and clear all progress."):
                    console.print("[cyan]Reset cancelled.[/cyan]")
                    return 0
            return self._reset_all_modules()

        # Get module number (prompt if not provided)
        module_number = args.module_number
        if not module_number:
            module_number = self._prompt_for_module()
            if not module_number:
                console.print("[cyan]Reset cancelled.[/cyan]")
                return 0

        # Validate module
        module_mapping = get_module_mapping()
        normalized = normalize_module_number(module_number)

        if normalized not in module_mapping:
            console.print(f"[red]Invalid module number: {module_number}[/red]")
            console.print(f"Available modules: 01-{max(module_mapping.keys())}")
            return 1

        module_name = module_mapping[normalized]

        # Confirm reset
        if not args.force:
            if not self._confirm_reset(f"This will reset module {normalized} ({module_name}) to its pristine state."):
                console.print("[cyan]Reset cancelled.[/cyan]")
                return 0

        return self._reset_single_module(module_number)
