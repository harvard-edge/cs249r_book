"""
Tiny🔥Torch Community Commands

Login, profile, and community status tools.
"""

from argparse import ArgumentParser, Namespace
from rich.panel import Panel
from rich.table import Table
from rich import box

from .base import BaseCommand
from .login import LoginCommand, LogoutCommand
from ..core import auth
from ..core.browser import open_url

# Community URLs
URL_COMMUNITY_MAP = "https://mlsysbook.ai/tinytorch/community/community.html"
URL_COMMUNITY_PROFILE = "https://mlsysbook.ai/tinytorch/community/?action=profile&community=true"


class CommunityCommand(BaseCommand):
    """Community commands - login, profile, map, and status."""

    @property
    def name(self) -> str:
        return "community"

    @property
    def description(self) -> str:
        return "Join the global community - login, profile, and map"

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

        # Profile command
        subparsers.add_parser(
            'profile',
            help='View/Edit your community profile'
        )

        # Status command
        subparsers.add_parser(
            'status',
            help='Show login status and user info'
        )

        # Map command
        subparsers.add_parser(
            'map',
            help='Open global community map'
        )

    def _show_status(self) -> int:
        """Show detailed auth status display."""
        is_logged_in = auth.is_logged_in()

        if is_logged_in:
            email = auth.get_user_email() or "Unknown Email"

            # Create an "ID Card" style display
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Field", style="dim")
            table.add_column("Value", style="bold")

            table.add_row("Status", "[green]● Online / Authenticated[/green]")
            table.add_row("User", f"[cyan]{email}[/cyan]")

            self.console.print(Panel(
                table,
                title="👤 TinyTorch Community ID",
                border_style="green",
                box=box.ROUNDED
            ))
        else:
            self.console.print(Panel(
                "[yellow]You are currently not logged in.[/yellow]\n\n"
                "To join the leaderboard and sync progress:\n"
                "  [bold green]tito login[/bold green]",
                title="❌ Not Authenticated",
                border_style="red",
                box=box.ROUNDED
            ))
        return 0

    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: login, logout, profile, status, map[/yellow]")
            return 1

        if args.community_command == 'login':
            return LoginCommand(self.config).run(args)
        elif args.community_command == 'logout':
            return LogoutCommand(self.config).run(args)
        elif args.community_command == 'profile':
            self.console.print("[cyan]Opening your profile...[/cyan]")
            open_url(URL_COMMUNITY_PROFILE, self.console)
            return 0
        elif args.community_command == 'map':
            self.console.print("[cyan]Opening community map...[/cyan]")
            open_url(URL_COMMUNITY_MAP, self.console)
            return 0
        elif args.community_command == 'status':
            return self._show_status()
        else:
            self.console.print(f"[red]❌ Unknown community command: {args.community_command}[/red]")
            return 1
