"""
Developer command group for TinyTorch CLI.

These commands are for TinyTorch developers and instructors, not students.
They help with:
- Pre-commit/pre-release verification (preflight)
- CI/CD integration
- Development workflows
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand
from .preflight import PreflightCommand
from .export import DevExportCommand


class DevCommand(BaseCommand):
    """Developer tools command group."""

    @property
    def name(self) -> str:
        return "dev"

    @property
    def description(self) -> str:
        return "Developer tools: preflight checks, CI/CD, workflows"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='dev_command',
            help='Developer subcommands',
            metavar='SUBCOMMAND'
        )

        # Preflight subcommand
        preflight_parser = subparsers.add_parser(
            'preflight',
            help='Run preflight verification checks before commit/release'
        )
        preflight_cmd = PreflightCommand(self.config)
        preflight_cmd.add_arguments(preflight_parser)

        # Export subcommand (rebuild curriculum from src/)
        export_parser = subparsers.add_parser(
            'export',
            help='Rebuild curriculum: src/*.py → modules/*.ipynb → tinytorch/core/*.py'
        )
        export_cmd = DevExportCommand(self.config)
        export_cmd.add_arguments(export_parser)

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'dev_command') or not args.dev_command:
            console.print(Panel(
                "[bold cyan]Developer Commands[/bold cyan]\n\n"
                "[bold]For developers and instructors - not for students.[/bold]\n\n"
                "Available subcommands:\n"
                "  • [bold]export[/bold]     - Rebuild curriculum from src/*.py files\n"
                "  • [bold]preflight[/bold]  - Run verification checks before commit/release\n\n"
                "[bold cyan]Export (Rebuild Curriculum):[/bold cyan]\n"
                "  [dim]tito dev export 01[/dim]           Export specific module\n"
                "  [dim]tito dev export 01 02 03[/dim]     Export multiple modules\n"
                "  [dim]tito dev export --all[/dim]        Export all modules\n"
                "  [bold red]⚠️  This OVERWRITES student notebooks![/bold red]\n\n"
                "[bold cyan]Preflight Levels:[/bold cyan]\n"
                "  [dim]tito dev preflight[/dim]           Standard checks (~30s)\n"
                "  [dim]tito dev preflight --quick[/dim]   Quick checks only (~10s)\n"
                "  [dim]tito dev preflight --full[/dim]    Full validation (~2-5min)\n"
                "  [dim]tito dev preflight --release[/dim] Release validation (~10-30min)\n\n"
                "[bold cyan]CI/CD Integration:[/bold cyan]\n"
                "  [dim]tito dev preflight --ci[/dim]      Non-interactive, exit codes\n"
                "  [dim]tito dev preflight --json[/dim]    JSON output for automation",
                title="🛠️ Developer Tools",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.dev_command == 'preflight':
            cmd = PreflightCommand(self.config)
            return cmd.execute(args)
        elif args.dev_command == 'export':
            cmd = DevExportCommand(self.config)
            return cmd.run(args)
        else:
            console.print(Panel(
                f"[red]Unknown dev subcommand: {args.dev_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
