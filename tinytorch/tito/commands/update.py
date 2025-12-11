"""
TinyTorch Update Command

Check for updates and download the latest version from GitHub.
Works with both git-based installs and standalone installs (no .git folder).
"""

import subprocess
import shutil
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import BaseCommand


class UpdateCommand(BaseCommand):
    """Check for and install TinyTorch updates."""

    REPO_URL = "https://github.com/harvard-edge/cs249r_book.git"
    SPARSE_PATH = "tinytorch"
    BRANCH = "main"

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

    def _has_git(self) -> bool:
        """Check if we're in a git repository."""
        git_dir = self.config.project_root / '.git'
        return git_dir.exists()

    def _get_current_version(self) -> str | None:
        """Get current version from pyproject.toml or version file."""
        try:
            pyproject = self.config.project_root / 'pyproject.toml'
            if pyproject.exists():
                content = pyproject.read_text()
                for line in content.split('\n'):
                    if line.strip().startswith('version'):
                        # Extract version from: version = "0.1.0"
                        return line.split('=')[1].strip().strip('"\'')
        except Exception:
            pass
        return None

    def _get_remote_version(self) -> str | None:
        """Fetch the latest version from GitHub."""
        try:
            # Create temp dir and do minimal fetch to get pyproject.toml
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone just enough to get version
                result = subprocess.run(
                    [
                        'git', 'clone', '--depth', '1', '--filter=blob:none',
                        '--sparse', '--branch', self.BRANCH,
                        self.REPO_URL, temp_dir
                    ],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return None

                # Set sparse checkout to only get pyproject.toml
                subprocess.run(
                    ['git', 'sparse-checkout', 'set', f'{self.SPARSE_PATH}/pyproject.toml'],
                    capture_output=True,
                    cwd=temp_dir
                )

                # Read version
                pyproject = Path(temp_dir) / self.SPARSE_PATH / 'pyproject.toml'
                if pyproject.exists():
                    content = pyproject.read_text()
                    for line in content.split('\n'):
                        if line.strip().startswith('version'):
                            return line.split('=')[1].strip().strip('"\'')
        except Exception:
            pass
        return None

    def _download_update(self) -> Path | None:
        """Download latest TinyTorch to a temp directory."""
        try:
            temp_dir = tempfile.mkdtemp(prefix='tinytorch_update_')

            self.console.print("[dim]  Downloading from GitHub...[/dim]")

            # Clone with sparse checkout
            result = subprocess.run(
                [
                    'git', 'clone', '--depth', '1', '--filter=blob:none',
                    '--sparse', '--branch', self.BRANCH,
                    self.REPO_URL, f'{temp_dir}/repo'
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.console.print(f"[red]  Failed to download: {result.stderr}[/red]")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            # Set sparse checkout to tinytorch folder
            subprocess.run(
                ['git', 'sparse-checkout', 'set', self.SPARSE_PATH],
                capture_output=True,
                cwd=f'{temp_dir}/repo'
            )

            return Path(temp_dir) / 'repo' / self.SPARSE_PATH

        except Exception as e:
            self.console.print(f"[red]  Download error: {e}[/red]")
            return None

    def _apply_update(self, new_source: Path, force: bool = False) -> bool:
        """Apply update by replacing files, preserving modules/ folder."""
        try:
            project_root = self.config.project_root

            # Folders to preserve (student work)
            preserve = ['modules', '.venv', '.tito', '.tinytorch']

            # Backup preserved folders
            self.console.print("[dim]  Backing up your work...[/dim]")
            backups = {}
            for folder in preserve:
                src = project_root / folder
                if src.exists():
                    backup_path = Path(tempfile.mkdtemp()) / folder
                    if src.is_dir():
                        shutil.copytree(src, backup_path)
                    else:
                        shutil.copy2(src, backup_path)
                    backups[folder] = backup_path

            # Remove old files (except preserved)
            self.console.print("[dim]  Updating files...[/dim]")
            for item in project_root.iterdir():
                if item.name not in preserve and item.name != '.git':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

            # Copy new files
            for item in new_source.iterdir():
                if item.name not in preserve:
                    dest = project_root / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

            # Restore preserved folders
            self.console.print("[dim]  Restoring your work...[/dim]")
            for folder, backup_path in backups.items():
                dest = project_root / folder
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                if backup_path.is_dir():
                    shutil.copytree(backup_path, dest)
                else:
                    shutil.copy2(backup_path, dest)
                # Clean up backup
                shutil.rmtree(backup_path.parent, ignore_errors=True)

            # Reinstall package
            self.console.print("[dim]  Reinstalling dependencies...[/dim]")
            subprocess.run(
                ['pip', 'install', '-e', '.', '-q'],
                cwd=project_root,
                capture_output=True
            )

            return True

        except Exception as e:
            self.console.print(f"[red]  Update error: {e}[/red]")
            return False

    def run(self, args: Namespace) -> int:
        """Execute update command."""
        from rich.panel import Panel

        self.console.print()
        self.console.print("[bold]🔄 Tiny🔥Torch Update[/bold]")
        self.console.print()

        # Get versions
        self.console.print("[dim]Checking for updates...[/dim]")
        current_version = self._get_current_version()
        remote_version = self._get_remote_version()

        if not remote_version:
            self.console.print("[red]❌ Could not check for updates.[/red]")
            self.console.print("[dim]Check your internet connection and try again.[/dim]")
            return 1

        # Compare versions
        if current_version == remote_version:
            self.console.print()
            self.console.print(Panel(
                f"[green]✓ TinyTorch is up to date![/green]\n\n"
                f"Version: [cyan]{current_version or 'unknown'}[/cyan]",
                border_style="green"
            ))
            return 0

        # Show available update
        self.console.print()
        self.console.print(Panel(
            f"[yellow]⬆ Update available![/yellow]\n\n"
            f"Current: [dim]{current_version or 'unknown'}[/dim]\n"
            f"Latest:  [green]{remote_version}[/green]",
            border_style="yellow"
        ))

        # If check-only mode, stop here
        if args.check:
            self.console.print()
            self.console.print("[dim]Run 'tito update' to install the update.[/dim]")
            return 0

        # Confirm update
        self.console.print()
        self.console.print("[dim]Your work in modules/ will be preserved.[/dim]")
        self.console.print()

        # Download and apply update
        self.console.print("[bold]Installing update...[/bold]")

        new_source = self._download_update()
        if not new_source:
            self.console.print("[red]❌ Download failed.[/red]")
            return 1

        if self._apply_update(new_source, force=args.force):
            # Clean up download
            shutil.rmtree(new_source.parent.parent, ignore_errors=True)

            self.console.print()
            self.console.print(Panel(
                f"[green]✓ TinyTorch updated successfully![/green]\n\n"
                f"Now at version: [cyan]{remote_version}[/cyan]\n\n"
                f"[dim]Your modules/ folder was preserved.[/dim]",
                border_style="green"
            ))
            return 0
        else:
            self.console.print()
            self.console.print("[red]❌ Update failed.[/red]")
            self.console.print("[dim]Your original files should still be intact.[/dim]")
            return 1
