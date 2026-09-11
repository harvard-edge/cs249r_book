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
import sys
import re
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple, List, Dict, Any

from ..base import BaseCommand


class UpdateCommand(BaseCommand):
    """Check for and install TinyTorch updates."""

    REPO_URL = "https://github.com/harvard-edge/cs249r_book.git"
    REPO = "harvard-edge/cs249r_book"
    TAGS_API = f"https://api.github.com/repos/{REPO}/tags?per_page=100"
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

    @staticmethod
    def _parse_version_tuple(v: str) -> Tuple[int, ...]:
        """Extract numeric components for semver comparison, e.g. '0.1.13' -> (0, 1, 13)."""
        nums = re.findall(r'\d+', v)
        return tuple(int(n) for n in nums) if nums else (0,)

    def _get_current_version(self) -> str:
        """Get current version from tinytorch package."""
        try:
            from tinytorch import __version__
            return __version__
        except ImportError:
            return "unknown"

    def _extract_best_tag(self, tags: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Find the semver-highest tinytorch-v* tag from a tag list."""
        candidates = []
        for tag in tags:
            tag_name = tag.get('name', '')
            if tag_name.startswith(self.TAG_PREFIX):
                version = tag_name[len(self.TAG_PREFIX):]
                parsed = self._parse_version_tuple(version)
                if parsed != (0,):
                    candidates.append((parsed, version, tag_name))

        if not candidates:
            return None, None

        # Sort descending by parsed version tuple
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]

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
                text=True, encoding="utf-8", errors="replace"
            )

            if result.returncode != 0:
                return self._get_latest_version_urllib()

            tags = json.loads(result.stdout)
            if not isinstance(tags, list):
                return self._get_latest_version_urllib()

            return self._extract_best_tag(tags)

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

            if not isinstance(tags, list):
                return None, None

            return self._extract_best_tag(tags)
        except Exception:
            return None, None

    def _compare_versions(self, current: str, latest: str) -> int:
        """
        Compare version strings.
        Returns: -1 if current < latest, 0 if equal, 1 if current > latest
        """
        try:
            current_parts = self._parse_version_tuple(current)
            latest_parts = self._parse_version_tuple(latest)

            if current_parts < latest_parts:
                return -1
            elif current_parts > latest_parts:
                return 1
            return 0
        except (ValueError, AttributeError):
            # If parsing fails, assume update needed if versions differ
            return -1 if current != latest else 0

    def _create_backup(self) -> Optional[Path]:
        """Create timestamped backup of student progress and work before updating."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = self.config.project_root / ".tito" / "backups" / f"pre_update_{timestamp}"
            backup_root.mkdir(parents=True, exist_ok=True)

            # Backup .tito state (progress, milestones, config)
            tito_dir = self.config.project_root / ".tito"
            for state_file in ["progress.json", "milestones.json", "config.json"]:
                src_file = tito_dir / state_file
                if src_file.exists():
                    shutil.copy2(src_file, backup_root / state_file)

            # Backup student modules
            student_modules = self.config.project_root / "modules"
            if student_modules.exists():
                shutil.copytree(student_modules, backup_root / "modules", dirs_exist_ok=True)

            # Backup student exported core
            student_core = self.config.project_root / "tinytorch" / "core"
            if student_core.exists():
                shutil.copytree(student_core, backup_root / "core", dirs_exist_ok=True)

            try:
                rel_path = backup_root.relative_to(self.config.project_root)
                self.console.print(f"[dim]  ✓ Snapshot created at: {rel_path}[/dim]")
            except ValueError:
                self.console.print(f"[dim]  ✓ Snapshot created at: {backup_root}[/dim]")
            return backup_root
        except Exception as e:
            self.console.print(f"[yellow]  ⚠ Could not create backup: {e}[/yellow]")
            return None

    def _download_latest(self, temp_dir: Path, tag_name: Optional[str] = None) -> bool:
        """
        Download latest TinyTorch to temp directory using git sparse checkout.
        Uses tag_name if provided, otherwise falls back to self.BRANCH.
        Returns True on success, False on failure.
        """
        try:
            repo_dir = temp_dir / "repo"
            clone_ref = tag_name or self.BRANCH

            # Clone with sparse checkout (minimal download)
            self.console.print(f"[dim]  Cloning repository (ref: {clone_ref})...[/dim]")
            result = subprocess.run(
                [
                    'git', 'clone',
                    '--depth', '1',
                    '--filter=blob:none',
                    '--sparse',
                    '--branch', clone_ref,
                    self.REPO_URL,
                    str(repo_dir)
                ],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace"
            )

            if result.returncode != 0 and clone_ref != self.BRANCH:
                # If clone by tag ref failed, try fallback to self.BRANCH
                self.console.print(f"[dim]  Ref clone failed, falling back to {self.BRANCH}...[/dim]")
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
                    text=True, encoding="utf-8", errors="replace"
                )

            if result.returncode != 0:
                self.console.print(f"[red]Git clone failed: {result.stderr}[/red]")
                return False

            # Set sparse checkout to only get tinytorch/
            self.console.print("[dim]  Fetching tinytorch files...[/dim]")
            result = subprocess.run(
                ['git', 'sparse-checkout', 'set', self.SPARSE_PATH],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
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
        """Update a directory safely with staging and rollback support."""
        try:
            if not src.exists():
                return True

            if name == "tito":
                # Do not delete the running CLI package directory; overwrite in-place
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                self.console.print(f"[dim]  ✓ Updated {name}/[/dim]")
                return True

            temp_backup = None
            if dst.exists():
                temp_backup = dst.with_name(f".{name}.bak_{os.getpid()}")
                if temp_backup.exists():
                    shutil.rmtree(temp_backup)
                dst.rename(temp_backup)

            try:
                shutil.copytree(src, dst)
                if temp_backup and temp_backup.exists():
                    shutil.rmtree(temp_backup)
                self.console.print(f"[dim]  ✓ Updated {name}/[/dim]")
                return True
            except Exception as copy_err:
                if temp_backup and temp_backup.exists():
                    if dst.exists():
                        shutil.rmtree(dst)
                    temp_backup.rename(dst)
                raise copy_err

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

    def _reinstall_package(self) -> bool:
        """Reinstall the TinyTorch package in editable mode after update."""
        try:
            self.console.print("[dim]  Reinstalling TinyTorch in development mode...[/dim]")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120
            )
            if result.returncode == 0:
                self.console.print("[dim]  ✓ Reinstalled TinyTorch package[/dim]")
                return True
            else:
                self.console.print(
                    f"[yellow]  ⚠ Warning: Package reinstall returned code {result.returncode}: {result.stderr.strip()}[/yellow]"
                )
                return False
        except Exception as e:
            self.console.print(f"[yellow]  ⚠ Warning: Could not reinstall package: {e}[/yellow]")
            return False

    def _verify_completed_modules(self) -> None:
        """Verify student's completed modules still import cleanly after an update."""
        progress_file = self.config.project_root / ".tito" / "progress.json"
        if not progress_file.exists():
            return

        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            completed = data.get("completed_modules", [])
            if not completed:
                return

            res = subprocess.run(
                [sys.executable, "-c", "import tinytorch; from tinytorch.core import *"],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=15
            )
            if res.returncode != 0:
                self.console.print()
                self.console.print("[yellow]⚠️  Notice: Student core code may need inspection after this update.[/yellow]")
                self.console.print("[dim]Run 'tito module status' to review your modules.[/dim]")
            else:
                self.console.print("[dim]  ✓ Verified core package imports[/dim]")
        except Exception:
            pass

    def _run_update(self, tag_name: Optional[str] = None) -> bool:
        """
        Perform in-place update while preserving student work.

        1. Create backup snapshot
        2. Download latest to temp directory
        3. Copy updateable directories/files
        4. Special handling for tinytorch/ package
        5. Reinstall pip package
        6. Verify completed modules
        """
        project_root = self.config.project_root
        success = True

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Create backup snapshot
            self.console.print()
            self.console.print("[bold]Creating backup snapshot...[/bold]")
            self._create_backup()

            # Step 2: Download latest
            self.console.print()
            self.console.print("[bold]Downloading latest version...[/bold]")
            if not self._download_latest(temp_path, tag_name=tag_name):
                return False

            # Source is the downloaded tinytorch/ subdirectory
            src_root = temp_path / "repo" / self.SPARSE_PATH

            if not src_root.exists():
                self.console.print("[red]Error: Downloaded files not found[/red]")
                return False

            # Step 3: Update directories
            self.console.print()
            self.console.print("[bold]Updating files...[/bold]")

            for dir_name in self.UPDATE_DIRS:
                src_dir = src_root / dir_name
                dst_dir = project_root / dir_name
                if not self._update_directory(src_dir, dst_dir, dir_name):
                    success = False

            # Step 4: Update individual files
            for file_name in self.UPDATE_FILES:
                src_file = src_root / file_name
                dst_file = project_root / file_name
                if not self._update_file(src_file, dst_file, file_name):
                    success = False

            # Step 5: Special handling for tinytorch/ package
            src_pkg = src_root / "tinytorch"
            dst_pkg = project_root / "tinytorch"
            if not self._update_tinytorch_package(src_pkg, dst_pkg):
                success = False

            # Step 6: Reinstall package in development mode
            self.console.print()
            self.console.print("[bold]Reinstalling dependencies...[/bold]")
            if not self._reinstall_package():
                success = False

            # Step 7: Verify completed modules
            self._verify_completed_modules()

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
        if self._run_update(tag_name=tag_name):
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
