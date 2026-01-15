"""
TinyTorch Update Command

Check for updates using GitHub API (fast) and download via curl.
Uses tinytorch-v* tags to determine latest version.
"""

from __future__ import annotations

import subprocess
import urllib.request
import json
import os
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

from ..base import BaseCommand


class UpdateCommand(BaseCommand):
    """Check for and install TinyTorch updates."""

    REPO = "harvard-edge/cs249r_book"
    TAGS_API = f"https://api.github.com/repos/{REPO}/tags"
    TAG_PREFIX = "tinytorch-v"
    INSTALL_URL = "https://mlsysbook.ai/tinytorch/install.sh"

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
            '--yes', '-y',
            action='store_true',
            help='Skip confirmation prompt'
        )

    def _get_current_version(self) -> str:
        """Get current version from tinytorch package."""
        try:
            from tinytorch import __version__
            return __version__
        except ImportError:
            return "unknown"

    def _get_latest_version(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch the latest tinytorch-v* tag from GitHub API.
        Returns (version_string, tag_name) or (None, None) on error.
        Uses curl for reliability across platforms (avoids SSL issues).
        """
        try:
            # Use curl for reliability (handles SSL better than urllib on macOS)
            result = subprocess.run(
                ['curl', '-fsSL', '--max-time', '10', self.TAGS_API],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return None, None
            
            tags = json.loads(result.stdout)
            
            # Find latest tinytorch-v* tag
            for tag in tags:
                tag_name = tag.get('name', '')
                if tag_name.startswith(self.TAG_PREFIX):
                    version = tag_name[len(self.TAG_PREFIX):]
                    return version, tag_name
            
            return None, None
            
        except json.JSONDecodeError:
            return None, None
        except FileNotFoundError:
            # curl not found, fall back to urllib
            return self._get_latest_version_urllib()
        except Exception as e:
            self.console.print(f"[dim]Error checking updates: {e}[/dim]")
            return None, None

    def _get_latest_version_urllib(self) -> Tuple[Optional[str], Optional[str]]:
        """Fallback using urllib if curl is not available."""
        try:
            import ssl
            # Create unverified context for macOS compatibility
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                self.TAGS_API,
                headers={'User-Agent': 'TinyTorch-CLI'}
            )
            
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                tags = json.loads(response.read().decode('utf-8'))
            
            for tag in tags:
                tag_name = tag.get('name', '')
                if tag_name.startswith(self.TAG_PREFIX):
                    version = tag_name[len(self.TAG_PREFIX):]
                    return version, tag_name
            
            return None, None
        except Exception:
            return None, None

    def _compare_versions(self, current: str, latest: str) -> int:
        """
        Compare version strings.
        Returns: -1 if current < latest, 0 if equal, 1 if current > latest
        """
        try:
            def parse_version(v: str) -> tuple:
                # Handle versions like "0.1.1" or "unknown"
                parts = v.split('.')
                return tuple(int(p) for p in parts if p.isdigit())
            
            current_parts = parse_version(current)
            latest_parts = parse_version(latest)
            
            if current_parts < latest_parts:
                return -1
            elif current_parts > latest_parts:
                return 1
            return 0
        except (ValueError, AttributeError):
            # If parsing fails, assume update needed if versions differ
            return -1 if current != latest else 0

    def _run_install(self) -> bool:
        """Run the install script to update TinyTorch."""
        try:
            self.console.print()
            self.console.print("[dim]Downloading and running installer...[/dim]")
            
            # Run curl | bash
            result = subprocess.run(
                ['bash', '-c', f'curl -fsSL {self.INSTALL_URL} | bash'],
                capture_output=False,  # Show output to user
                text=True
            )
            
            return result.returncode == 0
            
        except FileNotFoundError:
            self.console.print("[red]Error: bash or curl not found[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Install error: {e}[/red]")
            return False

    def run(self, args: Namespace) -> int:
        """Execute update command."""
        from rich.panel import Panel

        self.console.print()
        self.console.print("[bold]🔄 Tiny🔥Torch Update[/bold]")
        self.console.print()

        # Get current version
        current_version = self._get_current_version()
        self.console.print(f"[dim]Current version: v{current_version}[/dim]")

        # Check for updates
        self.console.print("[dim]Checking for updates...[/dim]")
        latest_version, tag_name = self._get_latest_version()

        if not latest_version:
            self.console.print()
            self.console.print("[red]❌ Could not check for updates[/red]")
            self.console.print("[dim]Check your internet connection and try again.[/dim]")
            return 1

        # Compare versions
        comparison = self._compare_versions(current_version, latest_version)

        if comparison >= 0:
            # Up to date or ahead
            self.console.print()
            self.console.print(Panel(
                f"[green]✅ You're on the latest version[/green]\n\n"
                f"Version: [cyan]v{current_version}[/cyan]",
                border_style="green"
            ))
            return 0

        # Update available
        self.console.print()
        self.console.print(Panel(
            f"[yellow]⬆️  Update available[/yellow]\n\n"
            f"Current: [dim]v{current_version}[/dim]\n"
            f"Latest:  [green]v{latest_version}[/green]",
            border_style="yellow"
        ))

        # If check-only mode, show install command and exit
        if args.check:
            self.console.print()
            self.console.print("To update, run:")
            self.console.print(f"  [cyan]tito system update[/cyan]")
            self.console.print()
            self.console.print("Or manually:")
            self.console.print(f"  [dim]curl -fsSL {self.INSTALL_URL} | bash[/dim]")
            return 0

        # Confirm update (unless --yes)
        if not args.yes:
            self.console.print()
            self.console.print(Panel(
                "[bold]This update reruns the installer and overwrites TinyTorch installation files.[/bold]\n"
                "[green]Your work in modules will be preserved.[/green]",
                title="Warning",
                border_style="yellow"
            ))
            self.console.print()
            try:
                response = input("Install update? [y/N] ").strip().lower()
                if response not in ('y', 'yes'):
                    self.console.print("[dim]Update cancelled.[/dim]")
                    return 0
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self.console.print("[dim]Update cancelled.[/dim]")
                return 0

        # Run update
        if self._run_install():
            self.console.print()
            self.console.print(Panel(
                f"[green]✅ TinyTorch updated successfully[/green]\n\n"
                f"Now at version: [cyan]v{latest_version}[/cyan]\n\n"
                f"[dim]Run 'tito --version' to verify.[/dim]",
                border_style="green"
            ))
            return 0
        else:
            self.console.print()
            self.console.print("[red]❌ Update failed[/red]")
            self.console.print("[dim]Try running manually:[/dim]")
            self.console.print(f"  curl -fsSL {self.INSTALL_URL} | bash")
            return 1
