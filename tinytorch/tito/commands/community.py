"""
Tiny🔥Torch Community Commands

Login and logout for the TinyTorch community.
"""

from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from .login import LoginCommand, LogoutCommand


class CommunityCommand(BaseCommand):
    """Community commands - login and logout."""

    @property
    def name(self) -> str:
        return "community"

    @property
    def description(self) -> str:
        return "Join the global community - login and connect with builders worldwide"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add community subcommands."""
        subparsers = parser.add_subparsers(
            dest='community_command',
            help='Community operations',
            metavar='COMMAND'
        )

        # Login command (delegates to LoginCommand)
        login_parser = subparsers.add_parser(
            'login',
            help='Log in to TinyTorch via web browser'
        )
        LoginCommand(self.config).add_arguments(login_parser)

        # Logout command (delegates to LogoutCommand)
        logout_parser = subparsers.add_parser(
            'logout',
            help='Log out of TinyTorch'
        )
        LogoutCommand(self.config).add_arguments(logout_parser)

    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: login, logout[/yellow]")
            return 1

        if args.community_command == 'login':
            return LoginCommand(self.config).run(args)
        elif args.community_command == 'logout':
            return LogoutCommand(self.config).run(args)
        else:
            self.console.print(f"[red]❌ Unknown community command: {args.community_command}[/red]")
            return 1
