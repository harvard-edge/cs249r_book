"""
TinyTorch CLI Main Entry Point

A professional command-line interface with proper architecture:
- Clean separation of concerns
- Proper error handling
- Logging support
- Configuration management
- Extensible command system
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Type, Optional, List

# Set TINYTORCH_QUIET before any tinytorch imports to suppress autograd messages
os.environ['TINYTORCH_QUIET'] = '1'

from .core.config import CLIConfig
from .core.virtual_env_manager import get_venv_path
from .core.console import get_console, print_banner, print_error, print_ascii_logo
from .core.exceptions import TinyTorchCLIError
from .core.theme import Theme
from rich.panel import Panel
from .commands.base import BaseCommand
from .commands.src import SrcCommand
from .commands.system import SystemCommand
from .commands.module import ModuleWorkflowCommand
from .commands.package import PackageCommand
from .commands.nbgrader import NBGraderCommand
from .commands.grade import GradeCommand
from .commands.logo import LogoCommand
from .commands.milestone import MilestoneCommand
from .commands.setup import SetupCommand
from .commands.benchmark import BenchmarkCommand
from .commands.community import CommunityCommand
from .commands.dev import DevCommand
from .commands.olympics import OlympicsCommand
from .commands.update import UpdateCommand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tito-cli.log'),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

class TinyTorchCLI:
    """Main CLI application class."""

    def __init__(self):
        """Initialize the CLI application."""
        self.config = CLIConfig.from_project_root()
        self.console = get_console()
        # SINGLE SOURCE OF TRUTH: All valid commands registered here
        self.commands: Dict[str, Type[BaseCommand]] = {
            # Essential
            'setup': SetupCommand,
            # Workflow (student-facing)
            'system': SystemCommand,
            'module': ModuleWorkflowCommand,
            # Developer tools
            'dev': DevCommand,
            'src': SrcCommand,
            'package': PackageCommand,
            'nbgrader': NBGraderCommand,
            # Progress tracking
            'milestone': MilestoneCommand,
            # Community
            'community': CommunityCommand,
            'benchmark': BenchmarkCommand,
            'olympics': OlympicsCommand,
            # Utilities
            'update': UpdateCommand,
            'grade': GradeCommand,
            'logo': LogoCommand,
        }

        # Command categorization for help display
        self.student_commands = ['module', 'milestone', 'community', 'benchmark', 'olympics']
        self.developer_commands = ['dev', 'system', 'src', 'package', 'nbgrader']

        # Welcome screen sections (used for both tito and tito --help)
        self.welcome_sections = {
            'quick_start': [
                (f'[{Theme.CAT_QUICKSTART}]tito setup[/{Theme.CAT_QUICKSTART}]', 'First-time setup (includes verification)'),
                (f'[{Theme.CAT_QUICKSTART}]tito module start 01[/{Theme.CAT_QUICKSTART}]', 'Start Module 01 (tensors)'),
                (f'[{Theme.CAT_QUICKSTART}]tito module complete 01[/{Theme.CAT_QUICKSTART}]', 'Test, export, and track progress'),
            ],
            'track_progress': [
                (f'[{Theme.CAT_PROGRESS}]tito module status[/{Theme.CAT_PROGRESS}]', 'View module progress'),
                (f'[{Theme.CAT_PROGRESS}]tito milestone status[/{Theme.CAT_PROGRESS}]', 'View unlocked capabilities'),
            ],
            'community': [
                (f'[{Theme.CAT_COMMUNITY}]tito community login[/{Theme.CAT_COMMUNITY}]', 'Log in to TinyTorch'),
                (f'[{Theme.CAT_COMMUNITY}]tito community logout[/{Theme.CAT_COMMUNITY}]', 'Log out of TinyTorch'),
            ],
            'help_docs': [
                (f'[{Theme.CAT_HELP}]tito system health[/{Theme.CAT_HELP}]', 'Check environment health'),
                (f'[{Theme.CAT_HELP}]tito --help[/{Theme.CAT_HELP}]', 'See all commands'),
            ]
        }

    def _generate_welcome_text(self) -> str:
        """Generate dynamic welcome text for interactive mode."""
        lines = []

        # Quick Start
        lines.append(f"[{Theme.SECTION}]Quick Start:[/{Theme.SECTION}]")
        for cmd, desc in self.welcome_sections['quick_start']:
            lines.append(f"  {cmd:<38} {desc}")

        # Track Progress
        lines.append(f"\n[{Theme.SECTION}]Track Progress:[/{Theme.SECTION}]")
        for cmd, desc in self.welcome_sections['track_progress']:
            lines.append(f"  {cmd:<38} {desc}")

        # Community
        lines.append(f"\n[{Theme.SECTION}]Community:[/{Theme.SECTION}]")
        for cmd, desc in self.welcome_sections['community']:
            lines.append(f"  {cmd:<38} {desc}")

        # Help & Docs
        lines.append(f"\n[{Theme.SECTION}]Help & Docs:[/{Theme.SECTION}]")
        for cmd, desc in self.welcome_sections['help_docs']:
            lines.append(f"  {cmd:<38} {desc}")

        return "\n".join(lines)

    def _generate_epilog(self) -> str:
        """Generate dynamic epilog from registered commands."""
        lines = []

        # Student Commands section
        lines.append("Student Commands:")
        for cmd_name in self.student_commands:
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name](self.config)
                # Simplify description for epilog (first sentence or shorter version)
                desc = cmd.description.split('.')[0].split('-')[0].strip()
                lines.append(f"  {cmd_name:<12} {desc}")
        lines.append("")

        # Developer Commands section
        lines.append("Developer Commands:")
        for cmd_name in self.developer_commands:
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name](self.config)
                desc = cmd.description.split('.')[0].split('-')[0].strip()
                lines.append(f"  {cmd_name:<12} {desc}")
        lines.append("")

        # Quick Start section (strip Rich formatting for plain text)
        lines.append("Quick Start:")
        for cmd, desc in self.welcome_sections['quick_start']:
            # Remove Rich color tags for plain epilog
            plain_cmd = cmd.replace('[green]', '').replace('[/green]', '')
            lines.append(f"  {plain_cmd:<28} {desc}")

        return "\n".join(lines)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="tito",
            description="Tiny🔥Torch CLI - Build ML systems from scratch",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._generate_epilog()
        )

        # Global options
        parser.add_argument(
            '--version',
            action='version',
            version='Tiny🔥Torch CLI 0.1.0'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--no-color',
            action='store_true',
            help='Disable colored output'
        )

        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )

        # Add command parsers
        for command_name, command_class in self.commands.items():
            # Create temporary instance to get metadata
            temp_command = command_class(self.config)
            cmd_parser = subparsers.add_parser(
                command_name,
                help=temp_command.description,
                formatter_class=argparse.RawDescriptionHelpFormatter
            )
            temp_command.add_arguments(cmd_parser)

        return parser

    def validate_environment(self) -> bool:
        """Validate the environment and show issues if any."""
        issues = self.config.validate(get_venv_path())

        if issues:
            print_error(
                "Environment validation failed:\n" + "\n".join(f"  • {issue}" for issue in issues),
                "Environment Issues"
            )
            self.console.print("\n[dim]Run 'tito system health' for detailed diagnosis[/dim]")
            # Return True to allow command execution despite validation issues
            # This is temporary for development
            return True

        return True

    def _show_help(self) -> int:
        """Show custom Rich-formatted help."""
        from rich.table import Table

        # Show ASCII logo
        print_ascii_logo()

        # Create commands table
        table = Table(show_header=True, header_style=Theme.SECTION, box=None, padding=(0, 2))
        table.add_column("Command", style=Theme.COMMAND, width=15)
        table.add_column("Description", style=Theme.DIM)

        # Add all commands dynamically
        for cmd_name, cmd_class in self.commands.items():
            cmd = cmd_class(self.config)
            table.add_row(cmd_name, cmd.description)

        self.console.print()
        self.console.print(f"[{Theme.SECTION}]Tiny🔥Torch CLI[/{Theme.SECTION}] - Build ML systems from scratch")
        self.console.print()
        self.console.print(f"[{Theme.EMPHASIS}]Usage:[/{Theme.EMPHASIS}] [{Theme.INFO}]tito[/{Theme.INFO}] [{Theme.OPTION}]COMMAND[/{Theme.OPTION}] [{Theme.DIM}][OPTIONS][/{Theme.DIM}]")
        self.console.print()
        self.console.print(f"[{Theme.SECTION}]Available Commands:[/{Theme.SECTION}]")
        self.console.print(table)
        self.console.print()
        self.console.print(self._generate_welcome_text())
        self.console.print()
        self.console.print(f"[{Theme.SECTION}]Global Options:[/{Theme.SECTION}]")
        self.console.print(f"  [{Theme.OPTION}]--help, -h[/{Theme.OPTION}]      Show this help message")
        self.console.print(f"  [{Theme.OPTION}]--version[/{Theme.OPTION}]       Show version number")
        self.console.print(f"  [{Theme.OPTION}]--verbose, -v[/{Theme.OPTION}]   Enable verbose output")
        self.console.print(f"  [{Theme.OPTION}]--no-color[/{Theme.OPTION}]      Disable colored output")
        self.console.print()

        return 0

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI application."""
        try:
            # Check for help flag before argparse to use Rich formatting
            if args and ('-h' in args or '--help' in args) and len(args) == 1:
                return self._show_help()

            parser = self.create_parser()
            parsed_args = parser.parse_args(args)

            # Update config with global options
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                self.config.verbose = True
                logging.getLogger().setLevel(logging.DEBUG)

            if hasattr(parsed_args, 'no_color') and parsed_args.no_color:
                self.config.no_color = True

            # Guard against running outside a virtual environment unless explicitly allowed
            if parsed_args.command not in ['setup', 'logo', None]:
                # Check both sys.prefix (traditional activation) and VIRTUAL_ENV (direnv/PATH-based)
                in_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV") is not None
                allow_system = os.environ.get("TITO_ALLOW_SYSTEM") == "1"
                if not in_venv and not allow_system:
                    print_error(
                        "TinyTorch must run inside a virtual environment.\n"
                        "Activate your project venv (for example, source .venv/bin/activate) "
                        "or set TITO_ALLOW_SYSTEM=1 to proceed at your own risk.",
                        "Virtual Environment Required"
                    )
                    return 1

            # Show banner for interactive commands (except logo which has its own display)
            # Skip banner for dev command with --json flag (CI/CD output)
            skip_banner = (
                parsed_args.command == 'logo' or
                (parsed_args.command == 'dev' and hasattr(parsed_args, 'json') and parsed_args.json)
            )
            if parsed_args.command and not self.config.no_color and not skip_banner:
                print_banner()

            # Validate environment for most commands (skip for health)
            skip_validation = (
                parsed_args.command in [None, 'version', 'help'] or
                (parsed_args.command == 'system' and
                 hasattr(parsed_args, 'system_command') and
                 parsed_args.system_command == 'health')
            )
            if not skip_validation:
                if not self.validate_environment():
                    return 1

            # Handle no command
            if not parsed_args.command:
                # Show ASCII logo first
                print_ascii_logo()

                # Generate dynamic welcome message
                self.console.print(Panel(
                    self._generate_welcome_text(),
                    title="Welcome to Tiny🔥Torch!",
                    border_style=Theme.BORDER_WELCOME
                ))
                return 0

            # Execute command
            if parsed_args.command in self.commands:
                command_class = self.commands[parsed_args.command]
                command = command_class(self.config)
                return command.execute(parsed_args)
            else:
                print_error(f"Unknown command: {parsed_args.command}")
                return 1

        except KeyboardInterrupt:
            self.console.print(f"\n[{Theme.WARNING}]Operation cancelled by user[/{Theme.WARNING}]")
            return 130
        except TinyTorchCLIError as e:
            logger.error(f"CLI error: {e}")
            print_error(str(e))
            return 1
        except Exception as e:
            logger.exception("Unexpected error in CLI")
            print_error(f"Unexpected error: {e}")
            return 1

def main() -> int:
    """Main entry point for the CLI."""
    cli = TinyTorchCLI()
    return cli.run(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(main())
