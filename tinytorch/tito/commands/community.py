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
from ..core.modules import get_module_mapping
from ..core.submission import (
    SubmissionHandler,
    set_auto_sync_consent,
    show_sync_disclosure,
    NO_SYNC_ENV,
)

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

        # Sync command
        sync_parser = subparsers.add_parser(
            'sync',
            help='Upload your local progress to the TinyTorch website'
        )
        auto_group = sync_parser.add_mutually_exclusive_group()
        auto_group.add_argument(
            '--enable-auto', action='store_true',
            help='Allow automatic sync after completing modules/milestones (saved on this machine)'
        )
        auto_group.add_argument(
            '--disable-auto', action='store_true',
            help=f'Turn off automatic sync (explicit "tito community sync" still works; {NO_SYNC_ENV}=1 also disables it)'
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
                "  [bold green]tito community login[/bold green]",
                title="❌ Not Authenticated",
                border_style="red",
                box=box.ROUNDED
            ))
        return 0

    def _sync(self, args: Namespace = None) -> int:
        """Upload local progress on demand.

        This is the explicit recovery path: a student who completed modules
        before logging in, or whose automatic sync was skipped, can run
        'tito community sync' to push their current progress.json at any time.
        """
        if getattr(args, 'enable_auto', False) or getattr(args, 'disable_auto', False):
            allowed = bool(getattr(args, 'enable_auto', False))
            if allowed:
                show_sync_disclosure(self.console)
            try:
                set_auto_sync_consent(allowed)
            except OSError as e:
                self.console.print(f"[red]Could not save sync preference: {e}[/red]")
                return 1
            state = "enabled" if allowed else "disabled"
            self.console.print(f"[green]Automatic sync {state}.[/green]")
            return 0

        if not auth.is_logged_in():
            self.console.print("[yellow]You are not logged in.[/yellow] Run [bold green]tito community login[/bold green] first, then sync.")
            return 1

        handler = SubmissionHandler(self.config, self.console)
        result = handler.sync_progress(total_modules=len(get_module_mapping()))
        return 0 if result.ok else 1

    def run(self, args: Namespace) -> int:
        """Execute community command."""
        if not args.community_command:
            self.console.print("[yellow]Please specify a community command: login, logout, profile, status, sync, map[/yellow]")
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
        elif args.community_command == 'sync':
            return self._sync(args)
        else:
            self.console.print(f"[red]❌ Unknown community command: {args.community_command}[/red]")
            return 1
