"""
TinyTorch Update Command

Check for updates using GitHub API and perform in-place updates.
Uses tinytorch-v* tags to determine latest version.

IMPORTANT: This command preserves student work during updates:
- modules/          (student notebooks in progress)
- tinytorch/core/   (student implementations)
- .tito/            (progress tracking)
- .venv/            (virtual environment)
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
import json
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, List

from ..base import BaseCommand


class UpdateCommand(BaseCommand):
    """Check for and install TinyTorch updates."""

    REPO_URL = "https://github.com/harvard-edge/cs249r_book.git"
    REPO = "harvard-edge/cs249r_book"
    TAGS_API = f"https://api.github.com/repos/{REPO}/tags"
    TAG_PREFIX = "tinytorch-v"
    BRANCH = "main"
    SPARSE_PATH = "tinytorch"

    # Directories/files to UPDATE (overwrite with new version)
    UPDATE_DIRS = [
        "src",           # Module source notebooks
        "tito",          # CLI tool
        "tests",         # Test suites
        "milestones",    # Milestone scripts
        "datasets",      # Sample datasets
        "bin",           # Entry point scripts
    ]
    UPDATE_FILES = [
        "requirements.txt",
        "pyproject.toml",
        "settings.ini",
        "README.md",
        "LICENSE",
    ]

    # Directories/files to PRESERVE (never overwrite)
    PRESERVE_DIRS = [
        "modules",       # Student work in progress
        ".venv",         # Virtual environment
        ".tito",         # Progress tracking
    ]
    PRESERVE_FILES = [
        "progress.json",  # Legacy progress file
    ]

    # Special handling for tinytorch/ package
    # We update __init__.py but preserve core/*.py (student implementations)

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
        import urllib.request
        import ssl

        try:
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

    def _download_latest(self, temp_dir: Path) -> bool:
        """
        Download latest TinyTorch to temp directory using git sparse checkout.
        Returns True on success, False on failure.
        """
        try:
            repo_dir = temp_dir / "repo"

            # Clone with sparse checkout (minimal download)
            self.console.print("[dim]  Cloning repository...[/dim]")
            result = subprocess.run(
                [
                    'git', 'clone',
                    '--depth', '1',
                    '--filter=blob:none',
                    '--sparse',
                    '--branch', self.BRANCH,
                    self.REPO_URL,
                    str(repo_dir)
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.console.print(f"[red]Git clone failed: {result.stderr}[/red]")
                return False

            # Set sparse checkout to only get tinytorch/
            self.console.print("[dim]  Fetching tinytorch files...[/dim]")
            result = subprocess.run(
                ['git', 'sparse-checkout', 'set', self.SPARSE_PATH],
                capture_output=True,
                text=True,
                cwd=repo_dir
            )

            if result.returncode != 0:
                self.console.print(f"[red]Sparse checkout failed: {result.stderr}[/red]")
                return False

            return True

        except FileNotFoundError:
            self.console.print("[red]Error: git not found[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Download error: {e}[/red]")
            return False

    def _update_directory(self, src: Path, dst: Path, name: str) -> bool:
        """Update a directory by replacing it entirely."""
        try:
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)
                self.console.print(f"[dim]  ✓ Updated {name}/[/dim]")
                return True
            return True  # Source doesn't exist is OK (optional dir)
        except Exception as e:
            self.console.print(f"[yellow]  ⚠ Could not update {name}/: {e}[/yellow]")
            return False

    def _update_file(self, src: Path, dst: Path, name: str) -> bool:
        """Update a single file."""
        try:
            if src.exists():
                shutil.copy2(src, dst)
                self.console.print(f"[dim]  ✓ Updated {name}[/dim]")
            return True
        except Exception as e:
            self.console.print(f"[yellow]  ⚠ Could not update {name}: {e}[/yellow]")
            return False

    def _update_tinytorch_package(self, src_pkg: Path, dst_pkg: Path) -> bool:
        """
        Update tinytorch/ package while preserving student implementations in core/.

        Updates:
        - __init__.py (version info)
        - Any new subpackages

        Preserves:
        - core/*.py (student implementations)
        """
        try:
            # Update __init__.py
            src_init = src_pkg / "__init__.py"
            dst_init = dst_pkg / "__init__.py"
            if src_init.exists():
                shutil.copy2(src_init, dst_init)
                self.console.print("[dim]  ✓ Updated tinytorch/__init__.py[/dim]")

            # Update core/__init__.py but NOT other .py files in core/
            src_core = src_pkg / "core"
            dst_core = dst_pkg / "core"
            if src_core.exists() and dst_core.exists():
                src_core_init = src_core / "__init__.py"
                dst_core_init = dst_core / "__init__.py"
                if src_core_init.exists():
                    shutil.copy2(src_core_init, dst_core_init)
                    self.console.print("[dim]  ✓ Updated tinytorch/core/__init__.py[/dim]")

            return True
        except Exception as e:
            self.console.print(f"[yellow]  ⚠ Could not update tinytorch package: {e}[/yellow]")
            return False

    def _run_update(self) -> bool:
        """
        Perform in-place update while preserving student work.

        1. Download latest to temp directory
        2. Copy updateable directories/files
        3. Special handling for tinytorch/ package
        4. Reinstall pip package
        """
        project_root = self.config.project_root
        success = True

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Download latest
            self.console.print()
            self.console.print("[bold]Downloading latest version...[/bold]")
            if not self._download_latest(temp_path):
                return False

            # Source is the downloaded tinytorch/ subdirectory
            src_root = temp_path / "repo" / self.SPARSE_PATH

            if not src_root.exists():
                self.console.print("[red]Error: Downloaded files not found[/red]")
                return False

            # Step 2: Update directories
            self.console.print()
            self.console.print("[bold]Updating files...[/bold]")

            for dir_name in self.UPDATE_DIRS:
                src_dir = src_root / dir_name
                dst_dir = project_root / dir_name
                if not self._update_directory(src_dir, dst_dir, dir_name):
                    success = False

            # Step 3: Update individual files
            for file_name in self.UPDATE_FILES:
                src_file = src_root / file_name
                dst_file = project_root / file_name
                if not self._update_file(src_file, dst_file, file_name):
                    success = False

            # Step 4: Special handling for tinytorch/ package
            src_pkg = src_root / "tinytorch"
            dst_pkg = project_root / "tinytorch"
            if not self._update_tinytorch_package(src_pkg, dst_pkg):
                success = False

        return success

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
            self.console.print("  [cyan]tito system update[/cyan]")
            return 0

        # Confirm update (unless --yes)
        if not args.yes:
            self.console.print()
            self.console.print(Panel(
                "[bold]This will update TinyTorch while preserving your work.[/bold]\n\n"
                "[green]Preserved:[/green] modules/, tinytorch/core/, progress\n"
                "[yellow]Updated:[/yellow] src/, tito/, tests/, milestones/",
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
        if self._run_update():
            self.console.print()
            self.console.print(Panel(
                f"[green]✅ TinyTorch updated successfully[/green]\n\n"
                f"Now at version: [cyan]v{latest_version}[/cyan]\n\n"
                f"[dim]Your work in modules/ was preserved.[/dim]",
                border_style="green"
            ))
            return 0
        else:
            self.console.print()
            self.console.print(Panel(
                "[yellow]⚠️  Update completed with some warnings[/yellow]\n\n"
                "[dim]Check the messages above for details.[/dim]",
                border_style="yellow"
            ))
            return 1
