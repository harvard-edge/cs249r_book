"""
Reset command for TinyTorch CLI: resets package and user data.
"""

import json
import shutil
from datetime import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from rich.panel import Panel
from rich.text import Text

from ..base import BaseCommand

class ResetCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset package files or user progress data"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='reset_command',
            help='Reset subcommands',
            metavar='SUBCOMMAND'
        )

        # Package reset (original functionality)
        package_parser = subparsers.add_parser(
            'package',
            help='Reset tinytorch package to clean state (remove exported files)'
        )
        package_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

        # All data reset
        all_parser = subparsers.add_parser(
            'all',
            help='Reset all user progress (modules + milestones + config)'
        )
        all_parser.add_argument("--backup", action="store_true", help="Create backup before reset")
        all_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

        # Progress reset
        progress_parser = subparsers.add_parser(
            'progress',
            help='Reset module completion tracking only'
        )
        progress_parser.add_argument("--backup", action="store_true", help="Create backup before reset")
        progress_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

        # Milestones reset
        milestones_parser = subparsers.add_parser(
            'milestones',
            help='Reset milestone achievements only'
        )
        milestones_parser.add_argument("--backup", action="store_true", help="Create backup before reset")
        milestones_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

        # Config reset
        config_parser = subparsers.add_parser(
            'config',
            help='Reset configuration to defaults'
        )
        config_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'reset_command') or not args.reset_command:
            console.print(Panel(
                "[bold cyan]Reset Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]package[/bold]     - Reset tinytorch package (remove exported files)\n"
                "  • [bold]all[/bold]         - Reset all user progress (modules + milestones + config)\n"
                "  • [bold]progress[/bold]    - Reset module completion tracking only\n"
                "  • [bold]milestones[/bold]  - Reset milestone achievements only\n"
                "  • [bold]config[/bold]      - Reset configuration to defaults\n\n"
                "[dim]Example: tito reset progress --backup[/dim]",
                title="Reset Command Group",
                border_style="bright_yellow"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.reset_command == 'package':
            return self._reset_package(args)
        elif args.reset_command == 'all':
            return self._reset_all(args)
        elif args.reset_command == 'progress':
            return self._reset_progress(args)
        elif args.reset_command == 'milestones':
            return self._reset_milestones(args)
        elif args.reset_command == 'config':
            return self._reset_config(args)
        else:
            console.print(Panel(
                f"[red]Unknown reset subcommand: {args.reset_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1

    def _reset_package(self, args: Namespace) -> int:
        """Reset tinytorch package (original functionality)."""
        console = self.console

        console.print(Panel("🔄 Resetting TinyTorch Package",
                           title="Package Reset", border_style="bright_yellow"))

        tinytorch_path = Path("tinytorch")

        if not tinytorch_path.exists():
            console.print(Panel("[yellow]⚠️  TinyTorch package directory not found. Nothing to reset.[/yellow]",
                              title="Nothing to Reset", border_style="yellow"))
            return 0

        # Ask for confirmation unless --force is used
        if not args.force:
            console.print("\n[yellow]This will remove all exported Python files from the tinytorch package.[/yellow]")
            console.print("[yellow]Notebooks in modules/ will be preserved.[/yellow]\n")

            try:
                response = input("Are you sure you want to reset? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                  title="Cancelled", border_style="cyan"))
                return 0

        reset_text = Text()
        reset_text.append("🗑️  Removing all exported files:\n", style="bold red")

        # Simple approach: remove all .py files except __init__.py
        files_removed = 0
        for py_file in tinytorch_path.rglob("*.py"):
            if py_file.name != "__init__.py":
                try:
                    rel_path = py_file.relative_to(tinytorch_path)
                    reset_text.append(f"  🗑️  tinytorch/{rel_path}\n", style="red")
                    py_file.unlink()
                    files_removed += 1
                except Exception as e:
                    reset_text.append(f"  ❌ Failed to remove {py_file}: {e}\n", style="red")

        # Remove __pycache__ directories
        for pycache in tinytorch_path.rglob("__pycache__"):
            if pycache.is_dir():
                reset_text.append(f"  🗑️  {pycache}/\n", style="red")
                shutil.rmtree(pycache)

        # Remove .pytest_cache if it exists
        pytest_cache = Path(".pytest_cache")
        if pytest_cache.exists():
            reset_text.append(f"  🗑️  .pytest_cache/\n", style="red")
            shutil.rmtree(pytest_cache)

        if files_removed > 0:
            reset_text.append(f"\n✅ Reset complete! Removed {files_removed} generated files.\n", style="bold green")
            reset_text.append("\n💡 Next steps:\n", style="bold yellow")
            reset_text.append("  • Run: tito module complete 01  - Re-export modules\n", style="white")

            console.print(Panel(reset_text, title="Reset Complete", border_style="green"))
        else:
            console.print(Panel("[yellow]No generated files found to remove.[/yellow]",
                              title="Nothing to Reset", border_style="yellow"))

        return 0

    def _create_backup(self) -> Path:
        """Create timestamped backup of .tito folder."""
        console = self.console
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f".tito_backup_{timestamp}")

        tito_dir = Path(".tito")
        if tito_dir.exists():
            shutil.copytree(tito_dir, backup_dir)
            console.print(f"[green]✅ Backup created: {backup_dir}[/green]")

        return backup_dir

    def _reset_all(self, args: Namespace) -> int:
        """Reset all user progress data."""
        console = self.console

        # Ask for confirmation
        if not args.force:
            console.print("\n[bold red]⚠️  Warning: This will reset ALL progress[/bold red]\n")
            console.print("[yellow]This will clear:[/yellow]")
            console.print("  • Module completion tracking")
            console.print("  • Milestone achievements")
            console.print("  • Configuration settings\n")
            console.print("[dim]Your code in modules/ will NOT be deleted.[/dim]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("\n[cyan]Reset cancelled.[/cyan]",
                                  title="Cancelled", border_style="cyan"))
                return 0

        # Create backup if requested
        if args.backup:
            self._create_backup()

        # Reset all data files
        tito_dir = Path(".tito")
        tito_dir.mkdir(parents=True, exist_ok=True)

        # Reset progress.json
        progress_file = tito_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "version": "1.0",
            "completed_modules": [],
            "completion_dates": {}
        }, indent=2))

        # Reset milestones.json
        milestones_file = tito_dir / "milestones.json"
        milestones_file.write_text(json.dumps({
            "version": "1.0",
            "completed_milestones": [],
            "completion_dates": {}
        }, indent=2))

        # Reset config.json
        config_file = tito_dir / "config.json"
        config_file.write_text(json.dumps({
            "logo_theme": "standard"
        }, indent=2))

        console.print(Panel(
            "[green]✅ All progress reset![/green]\n\n"
            "You're ready to start fresh.\\n"
            "Run: [cyan]tito module start 01[/cyan]",
            title="🔄 Reset Complete",
            border_style="green"
        ))

        return 0

    def _reset_progress(self, args: Namespace) -> int:
        """Reset module completion tracking only."""
        console = self.console

        # Ask for confirmation
        if not args.force:
            console.print("\n[bold yellow]⚠️  This will reset module completion tracking[/bold yellow]\n")
            console.print("[dim]Milestone achievements will be preserved.[/dim]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("\n[cyan]Reset cancelled.[/cyan]",
                                  title="Cancelled", border_style="cyan"))
                return 0

        # Create backup if requested
        if args.backup:
            self._create_backup()

        # Reset progress.json
        tito_dir = Path(".tito")
        tito_dir.mkdir(parents=True, exist_ok=True)

        progress_file = tito_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "version": "1.0",
            "completed_modules": [],
            "completion_dates": {}
        }, indent=2))

        console.print(Panel(
            "[green]✅ Module progress reset![/green]\n\n"
            "You can re-complete modules with:\\n"
            "[cyan]tito module complete XX[/cyan]",
            title="🔄 Progress Reset",
            border_style="green"
        ))

        return 0

    def _reset_milestones(self, args: Namespace) -> int:
        """Reset milestone achievements only."""
        console = self.console

        # Ask for confirmation
        if not args.force:
            console.print("\n[bold yellow]⚠️  This will reset milestone achievements[/bold yellow]\n")
            console.print("[dim]Module completion will be preserved.[/dim]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("\n[cyan]Reset cancelled.[/cyan]",
                                  title="Cancelled", border_style="cyan"))
                return 0

        # Create backup if requested
        if args.backup:
            self._create_backup()

        # Reset milestones.json
        tito_dir = Path(".tito")
        tito_dir.mkdir(parents=True, exist_ok=True)

        milestones_file = tito_dir / "milestones.json"
        milestones_file.write_text(json.dumps({
            "version": "1.0",
            "completed_milestones": [],
            "completion_dates": {}
        }, indent=2))

        console.print(Panel(
            "[green]✅ Milestone achievements reset![/green]\n\n"
            "You can re-run milestones with:\\n"
            "[cyan]tito milestone run XX[/cyan]",
            title="🔄 Milestones Reset",
            border_style="green"
        ))

        return 0

    def _reset_config(self, args: Namespace) -> int:
        """Reset configuration to defaults."""
        console = self.console

        # Ask for confirmation
        if not args.force:
            console.print("\n[bold yellow]⚠️  This will reset configuration to defaults[/bold yellow]\n")

            try:
                response = input("Continue? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    console.print(Panel("[cyan]Reset cancelled.[/cyan]",
                                      title="Cancelled", border_style="cyan"))
                    return 0
            except KeyboardInterrupt:
                console.print(Panel("\n[cyan]Reset cancelled.[/cyan]",
                                  title="Cancelled", border_style="cyan"))
                return 0

        # Reset config.json
        tito_dir = Path(".tito")
        tito_dir.mkdir(parents=True, exist_ok=True)

        config_file = tito_dir / "config.json"
        config_file.write_text(json.dumps({
            "logo_theme": "standard"
        }, indent=2))

        console.print(Panel(
            "[green]✅ Configuration reset to defaults![/green]",
            title="🔄 Config Reset",
            border_style="green"
        ))

        return 0
