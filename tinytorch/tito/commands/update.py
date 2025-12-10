"""
TinyTorch Update Command

Check for updates and pull the latest changes from the repository.
"""

import subprocess
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import BaseCommand


class UpdateCommand(BaseCommand):
    """Check for and install TinyTorch updates."""

    @property
    def name(self) -> str:
        return "update"

    @property
    def description(self) -> str:
        return "Check for and install updates"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add update subcommands."""
        parser.add_argument(
            '--check',
            action='store_true',
            help='Only check for updates, do not install'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if there are local changes'
        )

    def _get_git_root(self) -> Path | None:
        """Find the git root directory."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return None

    def _get_current_commit(self) -> str | None:
        """Get current commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None

    def _check_for_updates(self) -> tuple[bool, str | None, str | None]:
        """
        Check if updates are available.
        Returns: (has_updates, local_commit, remote_commit)
        """
        try:
            # Fetch latest from remote
            subprocess.run(
                ['git', 'fetch', 'origin'],
                capture_output=True,
                cwd=self.config.project_root
            )

            # Get local HEAD
            local_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )
            local_commit = local_result.stdout.strip()[:8] if local_result.returncode == 0 else None

            # Get remote HEAD (for the sparse checkout path)
            remote_result = subprocess.run(
                ['git', 'rev-parse', 'origin/main'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )
            remote_commit = remote_result.stdout.strip()[:8] if remote_result.returncode == 0 else None

            if local_commit and remote_commit:
                has_updates = local_commit != remote_commit
                return has_updates, local_commit, remote_commit

        except Exception as e:
            self.console.print(f"[red]Error checking for updates: {e}[/red]")

        return False, None, None

    def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _pull_updates(self, force: bool = False) -> bool:
        """Pull latest updates."""
        try:
            if force:
                # Stash local changes
                self.console.print("[dim]Stashing local changes...[/dim]")
                subprocess.run(
                    ['git', 'stash', 'push', '-m', 'tito-update-backup'],
                    capture_output=True,
                    cwd=self.config.project_root
                )

            # Pull updates
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=self.config.project_root
            )

            if result.returncode != 0:
                self.console.print(f"[red]Error pulling updates:[/red]")
                self.console.print(f"[dim]{result.stderr}[/dim]")
                return False

            if force:
                # Try to restore stashed changes
                self.console.print("[dim]Restoring local changes...[/dim]")
                subprocess.run(
                    ['git', 'stash', 'pop'],
                    capture_output=True,
                    cwd=self.config.project_root
                )

            return True

        except Exception as e:
            self.console.print(f"[red]Error pulling updates: {e}[/red]")
            return False

    def run(self, args: Namespace) -> int:
        """Execute update command."""
        from rich.panel import Panel

        self.console.print()
        self.console.print("[bold cyan]🔄 TinyTorch Update[/bold cyan]")
        self.console.print()

        # Check if we're in a git repo
        git_root = self._get_git_root()
        if not git_root:
            self.console.print("[red]❌ Not a git repository.[/red]")
            self.console.print("[dim]TinyTorch must be installed via git for updates to work.[/dim]")
            return 1

        # Check for updates
        self.console.print("[dim]Checking for updates...[/dim]")
        has_updates, local_commit, remote_commit = self._check_for_updates()

        if not has_updates:
            self.console.print()
            self.console.print(Panel(
                f"[green]✓ TinyTorch is up to date![/green]\n\n"
                f"Current version: [cyan]{local_commit or 'unknown'}[/cyan]",
                border_style="green"
            ))
            return 0

        # Show available update
        self.console.print()
        self.console.print(Panel(
            f"[yellow]⬆ Update available![/yellow]\n\n"
            f"Current: [dim]{local_commit}[/dim]\n"
            f"Latest:  [green]{remote_commit}[/green]",
            border_style="yellow"
        ))

        # If check-only mode, stop here
        if args.check:
            self.console.print()
            self.console.print("[dim]Run 'tito update' to install the update.[/dim]")
            return 0

        # Check for local changes
        has_local_changes = self._check_local_changes()
        if has_local_changes and not args.force:
            self.console.print()
            self.console.print("[yellow]⚠ You have local changes.[/yellow]")
            self.console.print("[dim]Your changes in modules/ will be preserved.[/dim]")
            self.console.print()
            self.console.print("Options:")
            self.console.print("  [cyan]tito update --force[/cyan]  - Stash changes, update, restore")
            self.console.print("  [cyan]git stash[/cyan]            - Manually stash first")
            return 1

        # Pull updates
        self.console.print()
        self.console.print("[dim]Installing update...[/dim]")

        if self._pull_updates(force=args.force):
            self.console.print()
            self.console.print(Panel(
                f"[green]✓ TinyTorch updated successfully![/green]\n\n"
                f"Now at version: [cyan]{remote_commit}[/cyan]",
                border_style="green"
            ))
            return 0
        else:
            self.console.print()
            self.console.print("[red]❌ Update failed. Please try again or update manually.[/red]")
            return 1
