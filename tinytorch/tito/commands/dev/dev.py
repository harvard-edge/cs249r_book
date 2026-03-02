"""
Developer command group for TinyTorch CLI.

These commands are for TinyTorch developers and instructors, not students.
Primary command: tito dev test (unified testing)
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel

from ..base import BaseCommand
from .test import DevTestCommand
from .export import DevExportCommand
from .build import DevBuildCommand
from .clean import DevCleanCommand


class DevCommand(BaseCommand):
    """Developer tools command group."""

    @property
    def name(self) -> str:
        return "dev"

    @property
    def description(self) -> str:
        return "Developer tools: test, export, build, clean"

    def add_arguments(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(
            dest='dev_command',
            help='Developer subcommands',
            metavar='SUBCOMMAND'
        )

        # Test subcommand (unified testing - primary command)
        test_parser = subparsers.add_parser(
            'test',
            help='Run tests: --unit, --integration, --e2e, --cli, --milestone, --all, --release'
        )
        test_cmd = DevTestCommand(self.config)
        test_cmd.add_arguments(test_parser)

        # Export subcommand (rebuild curriculum from src/)
        export_parser = subparsers.add_parser(
            'export',
            help='Rebuild curriculum: src/*.py → modules/*.ipynb → tinytorch/core/*.py'
        )
        export_cmd = DevExportCommand(self.config)
        export_cmd.add_arguments(export_parser)

        # Build subcommand (site, PDF, paper)
        build_parser = subparsers.add_parser(
            'build',
            help='Build site, PDF, or paper'
        )
        build_cmd = DevBuildCommand(self.config)
        build_cmd.add_arguments(build_parser)

        # Clean subcommand (remove build artifacts)
        clean_parser = subparsers.add_parser(
            'clean',
            help='Clean build artifacts'
        )
        clean_cmd = DevCleanCommand(self.config)
        clean_cmd.add_arguments(clean_parser)

    def run(self, args: Namespace) -> int:
        console = self.console

        if not hasattr(args, 'dev_command') or not args.dev_command:
            console.print(Panel(
                "[bold cyan]Developer Commands[/bold cyan]\n\n"
                "[bold]For developers and instructors - not for students.[/bold]\n\n"
                "[bold cyan]Testing (Primary):[/bold cyan]\n"
                "  [dim]tito dev test[/dim]               Run pytest unit tests (default)\n"
                "  [dim]tito dev test --all[/dim]         Run all test types\n"
                "  [dim]tito dev test --inline[/dim]      Inline tests from src/ (progressive)\n"
                "  [dim]tito dev test --unit[/dim]        Pytest unit tests\n"
                "  [dim]tito dev test --integration[/dim] Integration tests\n"
                "  [dim]tito dev test --e2e[/dim]         End-to-end tests\n"
                "  [dim]tito dev test --cli[/dim]         CLI tests\n"
                "  [dim]tito dev test --milestone[/dim]   Milestone script tests\n"
                "  [dim]tito dev test --release[/dim]     Full release validation\n"
                "  [dim]tito dev test --module 06[/dim]   Test specific module\n\n"
                "[bold cyan]Export (Rebuild Curriculum):[/bold cyan]\n"
                "  [dim]tito dev export --all[/dim]       Export all modules\n"
                "  [dim]tito dev export 01[/dim]          Export specific module\n"
                "  [bold red]⚠️  This OVERWRITES student notebooks![/bold red]\n\n"
                "[bold cyan]Build (Site & Paper):[/bold cyan]\n"
                "  [dim]tito dev build html[/dim]         Build HTML site\n"
                "  [dim]tito dev build serve[/dim]        Build and serve locally\n"
                "  [dim]tito dev build pdf[/dim]          Build PDF course guide\n"
                "  [dim]tito dev build paper[/dim]        Build research paper\n\n"
                "[bold cyan]Clean:[/bold cyan]\n"
                "  [dim]tito dev clean[/dim]              Clean all generated files\n"
                "  [dim]tito dev clean site[/dim]         Clean site build artifacts\n\n"
                "[bold cyan]CI/CD Integration:[/bold cyan]\n"
                "  [dim]tito dev test --ci[/dim]          JSON output for automation",
                title="🛠️ Developer Tools",
                border_style="bright_cyan"
            ))
            return 0

        # Execute the appropriate subcommand
        if args.dev_command == 'test':
            cmd = DevTestCommand(self.config)
            return cmd.run(args)
        elif args.dev_command == 'export':
            cmd = DevExportCommand(self.config)
            return cmd.run(args)
        elif args.dev_command == 'build':
            cmd = DevBuildCommand(self.config)
            return cmd.run(args)
        elif args.dev_command == 'clean':
            cmd = DevCleanCommand(self.config)
            return cmd.run(args)
        else:
            console.print(Panel(
                f"[red]Unknown dev subcommand: {args.dev_command}[/red]",
                title="Error",
                border_style="red"
            ))
            return 1
