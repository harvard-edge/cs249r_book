"""
Module Reset Command for TinyTorch CLI.

Provides comprehensive module reset functionality:
- Backup current work before reset
- Unexport from package
- Restore pristine source from git or backup
- Update progress tracking

This enables students to restart a module cleanly while preserving their work.
"""

import json
import shutil
import stat
import subprocess
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..base import BaseCommand


class ModuleResetCommand(BaseCommand):
    """Command to reset a module to clean state with backup functionality."""

    @property
    def name(self) -> str:
        return "reset"

    @property
    def description(self) -> str:
        return "Reset module to clean state (backup + unexport + restore)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add reset command arguments."""
        parser.add_argument(
            "module_number", nargs="?", default=None, help="Module number to reset (01, 02, etc.)"
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Reset all modules to pristine state",
        )
        parser.add_argument(
            "--soft",
            action="store_true",
            help="Soft reset: backup + restore source (keep package export)",
        )
        parser.add_argument(
            "--hard",
            action="store_true",
            help="Hard reset: backup + unexport + restore (full reset) [DEFAULT]",
        )
        parser.add_argument(
            "--from-git",
            action="store_true",
            help="Restore from git HEAD [DEFAULT]",
        )
        parser.add_argument(
            "--restore-backup",
            metavar="TIMESTAMP",
            help="Restore from specific backup timestamp",
        )
        parser.add_argument(
            "--list-backups", action="store_true", help="List available backups"
        )
        parser.add_argument(
            "--no-backup", action="store_true", help="Skip backup creation (dangerous)"
        )
        parser.add_argument(
            "--force", action="store_true", help="Skip confirmation prompts"
        )

    def get_module_mapping(self) -> Dict[str, str]:
        """Get mapping from numbers to module names."""
        return {
            "01": "01_tensor",
            "02": "02_activations",
            "03": "03_layers",
            "04": "04_losses",
            "05": "05_autograd",
            "06": "06_optimizers",
            "07": "07_training",
            "08": "08_dataloader",
            "09": "09_spatial",
            "10": "10_tokenization",
            "11": "11_embeddings",
            "12": "12_attention",
            "13": "13_transformers",
            "14": "14_profiling",
            "15": "15_quantization",
            "16": "16_acceleration",
            "17": "17_compression",
            "18": "18_memoization",
            "19": "19_benchmarking",
            "20": "20_capstone",
        }

    def normalize_module_number(self, module_input: str) -> str:
        """Normalize module input to 2-digit format."""
        if module_input.isdigit():
            return f"{int(module_input):02d}"
        return module_input

    def get_backup_dir(self) -> Path:
        """Get the backup directory, creating it if needed."""
        backup_dir = self.config.project_root / ".tito" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir

    def list_backups(self, module_name: str) -> List[Dict]:
        """List available backups for a module."""
        backup_dir = self.get_backup_dir()
        backups = []

        # Find all backup directories for this module
        pattern = f"{module_name}_*"
        for backup_path in backup_dir.glob(pattern):
            if backup_path.is_dir():
                # Read metadata if it exists
                metadata_file = backup_path / "backup_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            backups.append(
                                {
                                    "path": backup_path,
                                    "timestamp": metadata.get("timestamp"),
                                    "git_hash": metadata.get("git_hash"),
                                    "files": metadata.get("files", []),
                                }
                            )
                    except Exception:
                        # If metadata is corrupt, just use directory name
                        timestamp = backup_path.name.split("_", 1)[1]
                        backups.append(
                            {"path": backup_path, "timestamp": timestamp, "files": []}
                        )
                else:
                    # No metadata, use directory name
                    timestamp = backup_path.name.split("_", 1)[1]
                    backups.append(
                        {"path": backup_path, "timestamp": timestamp, "files": []}
                    )

        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    def show_backups_list(self, module_name: str) -> int:
        """Display list of available backups for a module."""
        console = self.console
        backups = self.list_backups(module_name)

        if not backups:
            console.print(
                Panel(
                    f"[yellow]No backups found for module: {module_name}[/yellow]",
                    title="No Backups",
                    border_style="yellow",
                )
            )
            return 0

        # Create table
        table = Table(title=f"Available Backups for {module_name}", show_header=True)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Git Hash", style="dim")
        table.add_column("Files", style="green")

        for backup in backups:
            table.add_row(
                backup["timestamp"],
                backup.get("git_hash", "unknown")[:8],
                str(len(backup.get("files", []))),
            )

        console.print(table)
        console.print(
            f"\n[dim]Restore a backup with:[/dim] [cyan]tito module reset {module_name} --restore-backup TIMESTAMP[/cyan]"
        )
        return 0

    def create_backup(self, module_name: str) -> Optional[Path]:
        """Create a backup of the current module state."""
        console = self.console

        # Get module directory
        module_dir = self.config.modules_dir / module_name
        if not module_dir.exists():
            console.print(
                f"[red]Module directory not found: {module_dir}[/red]"
            )
            return None

        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.get_backup_dir() / f"{module_name}_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Creating backup: {backup_dir.name}[/cyan]")

        # Get current git hash if in git repo
        git_hash = "unknown"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

        # Copy all Python files from module directory
        backed_up_files = []
        for py_file in module_dir.glob("*.py"):
            dest_file = backup_dir / py_file.name
            shutil.copy2(py_file, dest_file)
            backed_up_files.append(py_file.name)
            console.print(f"  [dim]✓ Backed up: {py_file.name}[/dim]")

        # Save metadata
        metadata = {
            "module_name": module_name,
            "timestamp": timestamp,
            "git_hash": git_hash,
            "files": backed_up_files,
            "backup_dir": str(backup_dir),
        }

        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[green]✓ Backup created successfully[/green]")
        return backup_dir

    def unexport_module(self, module_name: str) -> bool:
        """Remove module exports from the package."""
        console = self.console

        # Get export target from module's #| default_exp directive
        module_dir = self.config.modules_dir / module_name
        short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name
        dev_file = module_dir / f"{module_name}.py"

        if not dev_file.exists():
            console.print(f"[yellow]Dev file not found: {dev_file}[/yellow]")
            return True  # Nothing to unexport

        # Read export target
        export_target = None
        try:
            with open(dev_file, "r") as f:
                content = f.read()
                import re

                match = re.search(r"#\|\s*default_exp\s+([^\n\r]+)", content)
                if match:
                    export_target = match.group(1).strip()
        except Exception as e:
            console.print(f"[yellow]Could not read export target: {e}[/yellow]")
            return True

        if not export_target:
            console.print("[dim]No export target found (no #| default_exp)[/dim]")
            return True

        # Convert export target to file path
        target_file = (
            self.config.project_root
            / "tinytorch"
            / export_target.replace(".", "/")
        ).with_suffix(".py")

        if not target_file.exists():
            console.print(f"[dim]Export file not found (already removed?): {target_file}[/dim]")
            return True

        # Remove protection if file is read-only
        try:
            target_file.chmod(
                target_file.stat().st_mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            )
        except Exception:
            pass

        # Remove the exported file
        try:
            target_file.unlink()
            console.print(f"  [dim]✓ Removed export: {target_file.relative_to(self.config.project_root)}[/dim]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to remove export: {e}[/red]")
            return False

    def restore_from_git(self, module_name: str) -> bool:
        """Restore module from git HEAD."""
        console = self.console

        # Get module directory and dev file
        module_dir = self.config.modules_dir / module_name
        short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name
        dev_file = module_dir / f"{module_name}.py"

        console.print(f"[cyan]Restoring from git: {dev_file.relative_to(self.config.project_root)}[/cyan]")

        # Check if file exists in git
        try:
            result = subprocess.run(
                ["git", "ls-files", str(dev_file.relative_to(self.config.project_root))],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            if result.returncode != 0 or not result.stdout.strip():
                console.print(
                    f"[red]File not tracked in git: {dev_file}[/red]"
                )
                return False
        except Exception as e:
            console.print(f"[red]Git check failed: {e}[/red]")
            return False

        # Restore from git
        try:
            result = subprocess.run(
                ["git", "checkout", "HEAD", "--", str(dev_file.relative_to(self.config.project_root))],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            if result.returncode == 0:
                console.print(f"[green]✓ Restored from git HEAD[/green]")
                return True
            else:
                console.print(
                    f"[red]Git checkout failed: {result.stderr}[/red]"
                )
                return False
        except Exception as e:
            console.print(f"[red]Failed to restore from git: {e}[/red]")
            return False

    def restore_from_backup(self, module_name: str, timestamp: str) -> bool:
        """Restore module from a specific backup."""
        console = self.console

        # Find backup directory
        backup_dir = self.get_backup_dir() / f"{module_name}_{timestamp}"

        if not backup_dir.exists():
            console.print(
                f"[red]Backup not found: {backup_dir.name}[/red]"
            )
            return False

        # Get module directory
        module_dir = self.config.modules_dir / module_name

        console.print(f"[cyan]Restoring from backup: {backup_dir.name}[/cyan]")

        # Read metadata to get backed up files
        metadata_file = backup_dir / "backup_metadata.json"
        backed_up_files = []

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    backed_up_files = metadata.get("files", [])
            except Exception:
                pass

        # If no metadata, find all .py files in backup
        if not backed_up_files:
            backed_up_files = [f.name for f in backup_dir.glob("*.py")]

        # Restore each file
        restored_count = 0
        for filename in backed_up_files:
            backup_file = backup_dir / filename
            dest_file = module_dir / filename

            if backup_file.exists():
                try:
                    shutil.copy2(backup_file, dest_file)
                    console.print(f"  [dim]✓ Restored: {filename}[/dim]")
                    restored_count += 1
                except Exception as e:
                    console.print(
                        f"  [red]Failed to restore {filename}: {e}[/red]"
                    )
            else:
                console.print(
                    f"  [yellow]Backup file missing: {filename}[/yellow]"
                )

        if restored_count > 0:
            console.print(
                f"[green]✓ Restored {restored_count} file(s) from backup[/green]"
            )
            return True
        else:
            console.print("[red]Failed to restore any files from backup[/red]")
            return False

    def update_progress_tracking(self, module_name: str, module_number: str) -> None:
        """Update progress tracking to mark module as not completed."""
        console = self.console

        # Update progress.json (module_workflow.py format)
        progress_file = self.config.project_root / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    progress = json.load(f)

                # Remove from completed modules
                if "completed_modules" in progress:
                    if module_number in progress["completed_modules"]:
                        progress["completed_modules"].remove(module_number)
                        console.print(
                            f"  [dim]✓ Removed from completed modules[/dim]"
                        )

                # Update last_updated timestamp
                progress["last_updated"] = datetime.now().isoformat()

                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)
            except Exception as e:
                console.print(
                    f"[yellow]Could not update progress.json: {e}[/yellow]"
                )

        # Update .tito/progress.json (comprehensive format)
        tito_progress_dir = self.config.project_root / ".tito"
        tito_progress_file = tito_progress_dir / "progress.json"

        if tito_progress_file.exists():
            try:
                with open(tito_progress_file, "r") as f:
                    progress = json.load(f)

                # Remove from completed modules
                if "completed_modules" in progress:
                    if module_name in progress["completed_modules"]:
                        progress["completed_modules"].remove(module_name)

                # Remove completion date
                if "completion_dates" in progress:
                    if module_name in progress["completion_dates"]:
                        del progress["completion_dates"][module_name]

                with open(tito_progress_file, "w") as f:
                    json.dump(progress, f, indent=2)
            except Exception as e:
                console.print(
                    f"[yellow]Could not update .tito/progress.json: {e}[/yellow]"
                )

    def check_git_status(self) -> bool:
        """Check if there are uncommitted changes and warn user."""
        console = self.console

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )

            if result.returncode == 0 and result.stdout.strip():
                console.print(
                    Panel(
                        "[yellow]⚠️  You have uncommitted changes in your repository![/yellow]\n\n"
                        "[dim]Consider committing your work before resetting.[/dim]",
                        title="Uncommitted Changes",
                        border_style="yellow",
                    )
                )
                return False

            return True
        except Exception:
            # If git check fails, continue anyway
            return True

    def _reset_all_modules(self, args: Namespace) -> int:
        """Reset all modules to pristine state."""
        console = self.console
        
        module_mapping = self.get_module_mapping()
        
        # BIG WARNING
        console.print()
        console.print()
        console.print(
            Panel(
                f"[bold red]⚠️  WARNING: RESET ALL MODULES ⚠️[/bold red]\n\n"
                f"[bold yellow]This will:[/bold yellow]\n"
                f"  🗑️  Reset ALL {len(module_mapping)} modules to pristine state\n"
                f"  🗑️  Clear ALL progress tracking\n"
                f"  🗑️  Remove ALL package exports\n"
                f"  ♻️  Restore ALL source files from git\n"
                f"  📦 Re-export ALL modules fresh\n\n"
                f"[bold]Current state:[/bold]\n"
                f"  {'✅ Backups will be created' if not args.no_backup else '❌ NO BACKUPS - Your work will be LOST!'}\n\n"
                f"[dim]This is like a fresh install - use with caution![/dim]",
                title="⚠️  RESET ALL MODULES",
                border_style="red",
                padding=(1, 2),
            )
        )
        console.print()
        
        # Check git status
        self.check_git_status()
        
        # Confirmation (unless --force)
        if not args.force:
            console.print()
            console.print("[bold red]═══════════════════════════════════════════[/bold red]")
            console.print("[bold red]        CONFIRMATION REQUIRED              [/bold red]")
            console.print("[bold red]═══════════════════════════════════════════[/bold red]")
            console.print()
            
            if not args.no_backup:
                console.print("[green]✓ Your work will be backed up before reset[/green]")
            else:
                console.print("[bold red]✗ NO BACKUP - YOUR WORK WILL BE LOST![/bold red]")
            
            console.print()
            
            try:
                response = input("Type 'yes' to continue with reset (anything else cancels): ").strip().lower()
                if response != "yes":
                    console.print()
                    console.print(
                        Panel(
                            "[cyan]Reset cancelled. Your work is safe.[/cyan]",
                            title="✓ Cancelled",
                            border_style="cyan",
                        )
                    )
                    return 0
            except KeyboardInterrupt:
                console.print()
                console.print(
                    Panel(
                        "[cyan]Reset cancelled. Your work is safe.[/cyan]",
                        title="✓ Cancelled",
                        border_style="cyan",
                    )
                )
                return 0
        
        console.print()
        
        # Reset each module
        reset_count = 0
        failed_modules = []
        
        for module_num, module_name in sorted(module_mapping.items()):
            console.print(f"[cyan]Resetting {module_name}...[/cyan]")
            
            # Create backup if requested
            if not args.no_backup:
                backup_dir = self.create_backup(module_name)
                if not backup_dir:
                    console.print(f"[yellow]⚠ Backup failed for {module_name}[/yellow]")
            
            # Unexport
            self.unexport_module(module_name)
            
            # Restore from git
            if self.restore_from_git(module_name):
                console.print(f"[green]✓ {module_name} reset[/green]")
                reset_count += 1
            else:
                console.print(f"[red]✗ {module_name} failed[/red]")
                failed_modules.append(module_name)
            
            console.print()
        
        # Reset EVERYTHING for complete fresh install state
        console.print()
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print("[bold cyan]Resetting ALL data to fresh install state...[/bold cyan]")
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print()
        
        # Reset progress tracking
        console.print("[cyan]→ Module progress tracking...[/cyan]")
        for module_num in module_mapping.keys():
            self.update_progress_tracking(module_mapping[module_num], module_num)
        console.print("  [green]✓ Cleared[/green]")
        
        # Reset .tito/progress.json (comprehensive)
        tito_dir = self.config.project_root / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)
        
        console.print("[cyan]→ Progress files...[/cyan]")
        progress_file = tito_dir / "progress.json"
        progress_file.write_text(json.dumps({
            "version": "1.0",
            "completed_modules": [],
            "completion_dates": {}
        }, indent=2))
        console.print("  [green]✓ Cleared[/green]")
        
        # Reset milestones.json
        console.print("[cyan]→ Milestone achievements...[/cyan]")
        milestones_file = tito_dir / "milestones.json"
        milestones_file.write_text(json.dumps({
            "version": "1.0",
            "completed_milestones": [],
            "completion_dates": {}
        }, indent=2))
        console.print("  [green]✓ Cleared[/green]")
        
        # Reset config.json
        console.print("[cyan]→ Configuration settings...[/cyan]")
        config_file = tito_dir / "config.json"
        config_file.write_text(json.dumps({
            "logo_theme": "standard"
        }, indent=2))
        console.print("  [green]✓ Reset to defaults[/green]")
        console.print()
        
        # Re-export all modules to get fresh package files
        console.print("[cyan]Re-exporting modules to package...[/cyan]")
        try:
            result = subprocess.run(
                ["python", "-m", "nbdev.export"],
                capture_output=True,
                text=True,
                cwd=self.config.project_root,
            )
            if result.returncode == 0:
                console.print("[green]✓ Modules exported[/green]")
            else:
                console.print("[yellow]⚠ Export had issues (continuing)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not re-export: {e}[/yellow]")
        console.print()
        
        # Summary
        if failed_modules:
            console.print(
                Panel(
                    f"[yellow]⚠ Partial reset complete[/yellow]\n\n"
                    f"[green]Reset: {reset_count} modules[/green]\n"
                    f"[red]Failed: {len(failed_modules)} modules[/red]\n\n"
                    f"[dim]Failed modules: {', '.join(failed_modules)}[/dim]",
                    title="Reset Complete (with errors)",
                    border_style="yellow",
                )
            )
            return 1
        else:
            console.print(
                Panel(
                    f"[bold green]✅ COMPLETE RESET SUCCESSFUL![/bold green]\n\n"
                    f"[bold]What was reset:[/bold]\n"
                    f"  ✓ {reset_count} modules → pristine source files\n"
                    f"  ✓ All progress tracking → cleared\n"
                    f"  ✓ All milestones → cleared\n"
                    f"  ✓ Configuration → defaults\n"
                    f"  ✓ Package exports → re-exported fresh\n\n"
                    f"[bold cyan]🔥 You now have a completely fresh TinyTorch install![/bold cyan]\n\n"
                    f"[bold]Next steps:[/bold]\n"
                    f"  • [dim]tito module start 01[/dim] - Begin your journey\n"
                    f"  • [dim]tito module status[/dim] - Check status\n"
                    f"  • [dim]tito logo[/dim] - See what you're building",
                    title="🔥 Fresh Install State",
                    border_style="green",
                )
            )
            return 0

    def run(self, args: Namespace) -> int:
        """Execute the reset command."""
        console = self.console

        # Handle --all (reset all modules)
        if getattr(args, 'all', False):
            return self._reset_all_modules(args)

        # Handle --list-backups
        if getattr(args, 'list_backups', False):
            if not args.module_number:
                console.print(
                    "[red]Error: --list-backups requires a module number[/red]"
                )
                return 1

            module_mapping = self.get_module_mapping()
            normalized = self.normalize_module_number(args.module_number)

            if normalized not in module_mapping:
                console.print(f"[red]Invalid module number: {args.module_number}[/red]")
                return 1

            module_name = module_mapping[normalized]
            return self.show_backups_list(module_name)

        # Require module number
        if not args.module_number:
            console.print(
                Panel(
                    "[red]Error: Module number required[/red]\n\n"
                    "[dim]Examples:[/dim]\n"
                    "[dim]  tito module reset 01                    # Reset module 01[/dim]\n"
                    "[dim]  tito module reset 01 --list-backups     # Show backups[/dim]\n"
                    "[dim]  tito module reset 01 --soft             # Keep package export[/dim]\n"
                    "[dim]  tito module reset 01 --restore-backup   # Restore from backup[/dim]",
                    title="Module Number Required",
                    border_style="red",
                )
            )
            return 1

        # Normalize and validate module number
        module_mapping = self.get_module_mapping()
        normalized = self.normalize_module_number(args.module_number)

        if normalized not in module_mapping:
            console.print(f"[red]Invalid module number: {args.module_number}[/red]")
            console.print("Available modules: 01-20")
            return 1

        module_name = module_mapping[normalized]

        # Determine reset type
        is_hard_reset = args.hard or not args.soft  # Default to hard reset

        # Show reset plan
        console.print(
            Panel(
                f"[bold cyan]Module Reset: {module_name}[/bold cyan]\n\n"
                f"[bold]Reset Type:[/bold] {'Hard' if is_hard_reset else 'Soft'}\n"
                f"[bold]Actions:[/bold]\n"
                f"  {'✓' if not args.no_backup else '✗'} Backup current work\n"
                f"  {'✓' if is_hard_reset else '✗'} Unexport from package\n"
                f"  ✓ Restore pristine source\n"
                f"  ✓ Update progress tracking\n\n"
                f"[dim]{'Soft reset keeps package exports intact' if not is_hard_reset else 'Hard reset removes package exports'}[/dim]",
                title="Reset Plan",
                border_style="bright_yellow",
            )
        )

        # Check git status (warn but don't block)
        self.check_git_status()

        # Confirmation prompt (unless --force)
        if not args.force:
            console.print(
                "\n[yellow]This will reset the module to a clean state.[/yellow]"
            )
            if not args.no_backup:
                console.print("[green]Your current work will be backed up.[/green]")
            else:
                console.print(
                    "[red]Your current work will NOT be backed up![/red]"
                )

            try:
                response = input("\nContinue with reset? (y/N): ").strip().lower()
                if response not in ["y", "yes"]:
                    console.print(
                        Panel(
                            "[cyan]Reset cancelled.[/cyan]",
                            title="Cancelled",
                            border_style="cyan",
                        )
                    )
                    return 0
            except KeyboardInterrupt:
                console.print(
                    Panel(
                        "[cyan]Reset cancelled.[/cyan]",
                        title="Cancelled",
                        border_style="cyan",
                    )
                )
                return 0

        # Step 1: Create backup (unless --no-backup)
        if not args.no_backup:
            console.print("\n[bold]Step 1: Creating backup...[/bold]")
            backup_dir = self.create_backup(module_name)
            if not backup_dir:
                console.print("[red]Backup failed. Reset aborted.[/red]")
                return 1
        else:
            console.print(
                "\n[bold yellow]Step 1: Skipping backup (--no-backup)[/bold yellow]"
            )

        # Step 2: Unexport from package (unless --soft)
        if is_hard_reset:
            console.print("\n[bold]Step 2: Removing package exports...[/bold]")
            if not self.unexport_module(module_name):
                console.print(
                    "[yellow]Warning: Unexport may have failed (continuing)[/yellow]"
                )
        else:
            console.print(
                "\n[bold]Step 2: Keeping package exports (soft reset)[/bold]"
            )

        # Step 3: Restore source
        console.print("\n[bold]Step 3: Restoring pristine source...[/bold]")

        if args.restore_backup:
            # Restore from specific backup
            success = self.restore_from_backup(module_name, args.restore_backup)
        else:
            # Restore from git (default)
            success = self.restore_from_git(module_name)

        if not success:
            console.print("[red]Restore failed. Module may be in inconsistent state.[/red]")
            if not args.no_backup and 'backup_dir' in locals():
                console.print(
                    f"[yellow]Your work was backed up to: {backup_dir}[/yellow]"
                )
            return 1

        # Step 4: Update progress tracking
        console.print("\n[bold]Step 4: Updating progress tracking...[/bold]")
        self.update_progress_tracking(module_name, normalized)

        # Success summary
        console.print(
            Panel(
                f"[bold green]✓ Module {module_name} reset successfully![/bold green]\n\n"
                f"[green]Actions completed:[/green]\n"
                f"  {'✓ Work backed up' if not args.no_backup else '✗ No backup created'}\n"
                f"  {'✓ Package exports removed' if is_hard_reset else '✗ Package exports preserved'}\n"
                f"  ✓ Source restored to pristine state\n"
                f"  ✓ Progress tracking updated\n\n"
                f"[bold cyan]Next steps:[/bold cyan]\n"
                f"  • [dim]tito module start {normalized}[/dim]  - Begin working again\n"
                f"  • [dim]tito module resume {normalized}[/dim] - Continue from where you left off\n"
                + (
                    f"  • [dim]tito module reset {normalized} --list-backups[/dim] - View backups\n"
                    if not args.no_backup
                    else ""
                ),
                title="Reset Complete",
                border_style="green",
            )
        )

        return 0
