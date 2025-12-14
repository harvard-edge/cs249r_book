"""
Package command group for TinyTorch CLI: nbdev integration and package management.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand
from .reset import ResetCommand
from .nbdev import NbdevCommand

class PackageCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "package"

    @property
    def description(self) -> str:
        return "Package management and nbdev integration commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='package_command',
            help='Package subcommands',
            metavar='SUBCOMMAND'
        )

        # Reset subcommand
        reset_parser = subparsers.add_parser(
            'reset',
            help='Reset tinytorch package to clean state'
        )
        reset_cmd = ResetCommand(self.config)
        reset_cmd.add_arguments(reset_parser)

        # Nbdev subcommand
        nbdev_parser = subparsers.add_parser(
            'nbdev',
            help='nbdev notebook development commands'
        )
        nbdev_cmd = NbdevCommand(self.config)
        nbdev_cmd.add_arguments(nbdev_parser)

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'package_command') or not args.package_command:
            console.print(Panel(
                "[bold cyan]Package Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]reset[/bold]   - Reset tinytorch package to clean state\n"
                "  • [bold]nbdev[/bold]   - nbdev notebook development commands\n\n"
                "[dim]Examples:[/dim]\n"
                "[dim]  tito package reset --force[/dim]\n"
                "[dim]  tito package nbdev --export[/dim]\n\n"
                "[dim]Note: Use 'tito module export' for exporting modules[/dim]",
                title="Package Command Group",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.package_command == 'reset':
            cmd = ResetCommand(self.config)
            return cmd.execute(args)
        elif args.package_command == 'nbdev':
            cmd = NbdevCommand(self.config)
            return cmd.execute(args)
        else:
            console.print(Panel(
                f"[red]Unknown package subcommand: {args.package_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
