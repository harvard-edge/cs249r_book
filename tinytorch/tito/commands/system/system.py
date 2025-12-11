"""
System command group for TinyTorch CLI: environment, configuration, and system tools.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand
from .info import InfoCommand
from .health import HealthCommand
from .jupyter import JupyterCommand

class SystemCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "System environment and configuration commands"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='system_command',
            help='System subcommands',
            metavar='SUBCOMMAND'
        )

        # Info subcommand
        info_parser = subparsers.add_parser(
            'info',
            help='Show system and environment information'
        )
        info_cmd = InfoCommand(self.config)
        info_cmd.add_arguments(info_parser)

        # Health subcommand
        health_parser = subparsers.add_parser(
            'health',
            help='Environment health check and validation'
        )
        health_cmd = HealthCommand(self.config)
        health_cmd.add_arguments(health_parser)

        # Jupyter subcommand
        jupyter_parser = subparsers.add_parser(
            'jupyter',
            help='Start Jupyter notebook server'
        )
        jupyter_cmd = JupyterCommand(self.config)
        jupyter_cmd.add_arguments(jupyter_parser)

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'system_command') or not args.system_command:
            console.print(Panel(
                "[bold cyan]System Commands[/bold cyan]\n\n"
                "Available subcommands:\n"
                "  • [bold]info[/bold]    - Show system/environment information\n"
                "  • [bold]health[/bold]  - Environment health check and validation\n"
                "  • [bold]jupyter[/bold] - Start Jupyter notebook server\n\n"
                "[dim]Example: tito system health[/dim]",
                title="System Command Group",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.system_command == 'info':
            cmd = InfoCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'health':
            cmd = HealthCommand(self.config)
            return cmd.execute(args)
        elif args.system_command == 'jupyter':
            cmd = JupyterCommand(self.config)
            return cmd.execute(args)
        else:
            console.print(Panel(
                f"[red]Unknown system subcommand: {args.system_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
