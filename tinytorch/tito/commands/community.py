"""
Tiny🔥Torch Community Commands

Login, logout, and connect with the TinyTorch community.
"""

from argparse import ArgumentParser, Namespace

from rich.panel import Panel

from .base import BaseCommand
from .login import LoginCommand, LogoutCommand


class CommunityCommand(BaseCommand):
    """Community commands - login, logout, leaderboard, and compete."""

    @property
    def name(self) -> str:
        return "community"

    @property
    def description(self) -> str:
        return "Join the global community - connect with builders worldwide"

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

        # Leaderboard command (coming soon)
        subparsers.add_parser(
            'leaderboard',
            help='View global leaderboard (coming soon)'
        )

        # Compete command (coming soon)
        subparsers.add_parser(
            'compete',
            help='Join competitions and challenges (coming soon)'
        )

    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: login, logout, leaderboard, compete[/yellow]")
            return 1

        if args.community_command == 'login':
            return LoginCommand(self.config).run(args)
        elif args.community_command == 'logout':
            return LogoutCommand(self.config).run(args)
        elif args.community_command == 'leaderboard':
            return self._show_coming_soon("Leaderboard", "Compare your optimizations with learners worldwide")
        elif args.community_command == 'compete':
            return self._show_coming_soon("Competitions", "Join TinyML optimization challenges")
        else:
            self.console.print(f"[red]❌ Unknown community command: {args.community_command}[/red]")
            return 1

    def _show_coming_soon(self, feature: str, description: str) -> int:
        """Show coming soon message for a feature."""
        self.console.print(Panel(
            f"[bold yellow]🚧 {feature} - Coming Soon![/bold yellow]\n\n"
            f"[dim]{description}[/dim]\n\n"
            "This feature is currently under development.\n\n"
            "In the meantime:\n"
            "  • Complete modules to build your skills\n"
            "  • Track your progress with [cyan]tito milestones status[/cyan]\n"
            "  • Join the community with [cyan]tito community login[/cyan]",
            title=f"🎯 {feature}",
            border_style="yellow"
        ))
        return 0
