#!/usr/bin/env python3
"""
Git Cleanup Tool - Find and Remove Latest Files with History Reset
A powerful tool to identify recently modified files and permanently remove them from git history
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import shutil

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import print as rprint

console = Console()

class GitCleanupTool:
    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.console = Console()

    def run_git_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run a git command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr if e.stderr else str(e)

    def check_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        success, _ = self.run_git_command(["git", "rev-parse", "--git-dir"])
        return success

    def get_recent_files(self, days: int = 7, max_files: int = 50) -> List[Dict]:
        """Get files modified in the last N days"""
        console.print(f"[blue]🔍 Finding files modified in the last {days} days...[/blue]")

        # Get files modified in the last N days
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        cmd = [
            "git", "log", "--since", since_date,
            "--name-only", "--pretty=format:",
            "--diff-filter=M"  # Only modified files
        ]

        success, output = self.run_git_command(cmd)
        if not success:
            console.print(f"[red]❌ Error getting recent files: {output}[/red]")
            return []

        # Parse the output to get unique files
        files = set()
        for line in output.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('commit') and not line.startswith('Author'):
                files.add(line)

        # Get additional info for each file
        file_info = []
        for file_path in sorted(files)[:max_files]:
            if os.path.exists(os.path.join(self.repo_path, file_path)):
                file_info.append({
                    'path': file_path,
                    'size': os.path.getsize(os.path.join(self.repo_path, file_path)),
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(self.repo_path, file_path))
                    ).strftime("%Y-%m-%d %H:%M")
                })

        return file_info

    def get_large_files(self, min_size_mb: int = 10) -> List[Dict]:
        """Find large files in the repository"""
        console.print(f"[blue]🔍 Finding files larger than {min_size_mb}MB...[/blue]")

        # Alternative approach using find
        find_cmd = [
            "find", ".", "-type", "f", "-size", f"+{min_size_mb}M",
            "-not", "-path", "./.git/*"
        ]

        success, output = self.run_git_command(find_cmd)
        if not success:
            console.print(f"[red]❌ Error finding large files: {output}[/red]")
            return []

        file_info = []
        for line in output.strip().split('\n'):
            if line.strip():
                file_path = line.strip()
                if os.path.exists(file_path):
                    file_info.append({
                        'path': file_path,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).strftime("%Y-%m-%d %H:%M")
                    })

        return file_info

    def get_git_history_size(self) -> Dict:
        """Get git repository size information"""
        console.print("[blue]📊 Analyzing git repository size...[/blue]")

        # Get total size
        cmd = ["git", "count-objects", "-vH"]
        success, output = self.run_git_command(cmd)

        size_info = {}
        if success:
            for line in output.strip().split('\n'):
                if 'size-pack' in line:
                    size_info['pack_size'] = line.split(':')[1].strip()
                elif 'size-garbage' in line:
                    size_info['garbage_size'] = line.split(':')[1].strip()

        # Get number of commits
        cmd = ["git", "rev-list", "--count", "HEAD"]
        success, output = self.run_git_command(cmd)
        if success:
            size_info['total_commits'] = int(output.strip())

        return size_info

    def display_files_table(self, files: List[Dict], title: str):
        """Display files in a beautiful table"""
        if not files:
            console.print(f"[yellow]⚠️  No files found for: {title}[/yellow]")
            return

        table = Table(title=title, show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("File Path", style="cyan", width=50)
        table.add_column("Size", style="green", width=12)
        table.add_column("Modified", style="yellow", width=20)

        for i, file_info in enumerate(files, 1):
            size_mb = file_info['size'] / (1024 * 1024)
            size_str = f"{size_mb:.1f}MB" if size_mb >= 1 else f"{file_info['size'] / 1024:.1f}KB"

            table.add_row(
                str(i),
                file_info['path'][:48] + "..." if len(file_info['path']) > 48 else file_info['path'],
                size_str,
                file_info['modified']
            )

        console.print(table)
        return files

    def select_files_to_delete(self, files: List[Dict]) -> List[str]:
        """Interactive file selection for deletion"""
        if not files:
            return []

        console.print("\n[bold yellow]🗑️  Select files to delete:[/bold yellow]")
        console.print("[dim]Enter file numbers separated by commas (e.g., 1,3,5)[/dim]")
        console.print("[dim]Or enter 'all' to select all files[/dim]")
        console.print("[dim]Or enter 'none' to skip[/dim]")

        while True:
            try:
                selection = Prompt.ask("File numbers", default="none")

                if selection.lower() == "none":
                    return []
                elif selection.lower() == "all":
                    return [f['path'] for f in files]
                else:
                    # Parse comma-separated numbers
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_files = []

                    for idx in indices:
                        if 0 <= idx < len(files):
                            selected_files.append(files[idx]['path'])
                        else:
                            console.print(f"[red]❌ Invalid file number: {idx + 1}[/red]")

                    if selected_files:
                        return selected_files
                    else:
                        console.print("[red]❌ No valid files selected[/red]")

            except ValueError:
                console.print("[red]❌ Invalid input. Please enter numbers separated by commas.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 Selection cancelled[/yellow]")
                return []

    def backup_files(self, files: List[str]) -> str:
        """Create a backup of files before deletion"""
        backup_dir = self.repo_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)

        console.print(f"[blue]💾 Creating backup in: {backup_dir}[/blue]")

        with Progress() as progress:
            task = progress.add_task("Backing up files...", total=len(files))

            for file_path in files:
                try:
                    source = self.repo_path / file_path
                    if source.exists():
                        # Create directory structure in backup
                        backup_file = backup_dir / file_path
                        backup_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, backup_file)
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not backup {file_path}: {e}[/yellow]")

                progress.advance(task)

        return str(backup_dir)

    def remove_files_from_git(self, files: List[str], method: str = "filter-branch") -> bool:
        """Remove files from git history using specified method"""
        console.print(f"[red]🗑️  Removing {len(files)} files from git history using {method}...[/red]")

        if method == "filter-branch":
            return self._remove_with_filter_branch(files)
        elif method == "bfg":
            return self._remove_with_bfg(files)
        else:
            console.print(f"[red]❌ Unknown method: {method}[/red]")
            return False

    def _remove_with_filter_branch(self, files: List[str]) -> bool:
        """Remove files using git filter-branch"""
        # Create a script to remove files
        script_content = "#!/bin/bash\n"
        for file_path in files:
            script_content += f'git rm --cached --ignore-unmatch "{file_path}"\n'

        script_path = self.repo_path / "remove_files.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)

        try:
            # Run filter-branch
            cmd = [
                "git", "filter-branch", "--force", "--index-filter",
                f"'{script_path}'", "--prune-empty", "--tag-name-filter", "cat", "--", "--all"
            ]

            console.print("[yellow]⚠️  This will rewrite git history. Make sure you have a backup![/yellow]")
            if not Confirm.ask("Continue with filter-branch?"):
                return False

            success, output = self.run_git_command(cmd, capture_output=False)

            if success:
                # Clean up
                script_path.unlink()

                # Force garbage collection
                self.run_git_command(["git", "for-each-ref", "--format='delete %(refname)'", "refs/original"])
                self.run_git_command(["git", "reflog", "expire", "--expire=now", "--all"])
                self.run_git_command(["git", "gc", "--prune=now", "--aggressive"])

                console.print("[green]✅ Files removed from git history successfully[/green]")
                return True
            else:
                console.print(f"[red]❌ Error removing files: {output}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Error during filter-branch: {e}[/red]")
            return False
        finally:
            if script_path.exists():
                script_path.unlink()

    def _remove_with_bfg(self, files: List[str]) -> bool:
        """Remove files using BFG Repo-Cleaner"""
        console.print("[yellow]⚠️  BFG method requires BFG Repo-Cleaner to be installed[/yellow]")
        console.print("[dim]Install with: brew install bfg (macOS) or download from https://rtyley.github.io/bfg-repo-cleaner/[/dim]")

        if not Confirm.ask("Continue with BFG method?"):
            return False

        # Create a file list for BFG
        file_list = self.repo_path / "files_to_delete.txt"
        with open(file_list, 'w') as f:
            for file_path in files:
                f.write(f"{file_path}\n")

        try:
            # Run BFG
            cmd = ["bfg", "--delete-files", "files_to_delete.txt"]
            success, output = self.run_git_command(cmd, capture_output=False)

            if success:
                # Clean up
                file_list.unlink()

                # Force garbage collection
                self.run_git_command(["git", "reflog", "expire", "--expire=now", "--all"])
                self.run_git_command(["git", "gc", "--prune=now", "--aggressive"])

                console.print("[green]✅ Files removed from git history successfully[/green]")
                return True
            else:
                console.print(f"[red]❌ Error removing files: {output}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Error during BFG cleanup: {e}[/red]")
            return False
        finally:
            if file_list.exists():
                file_list.unlink()

    def show_repository_stats(self):
        """Show repository statistics"""
        console.print("[bold blue]📊 Repository Statistics[/bold blue]")

        # Get basic stats
        size_info = self.get_git_history_size()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        if size_info:
            for key, value in size_info.items():
                table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)

    def interactive_cleanup(self):
        """Interactive cleanup workflow"""
        console.print(Panel.fit(
            "[bold red]🗑️  Git Cleanup Tool[/bold red]\n"
            "[dim]Find and permanently remove files from git history[/dim]",
            border_style="red"
        ))

        # Check if we're in a git repo
        if not self.check_git_repo():
            console.print("[red]❌ Not a git repository. Please run this from a git repo.[/red]")
            return

        # Show repository stats
        self.show_repository_stats()

        # Get recent files
        days = Prompt.ask("Days to look back", default="7")
        try:
            days = int(days)
        except ValueError:
            days = 7

        recent_files = self.get_recent_files(days=days)
        self.display_files_table(recent_files, f"Files Modified in Last {days} Days")

        # Get large files
        min_size = Prompt.ask("Minimum file size (MB)", default="10")
        try:
            min_size = int(min_size)
        except ValueError:
            min_size = 10

        large_files = self.get_large_files(min_size_mb=min_size)
        self.display_files_table(large_files, f"Files Larger Than {min_size}MB")

        # Combine and deduplicate files
        all_files = recent_files + large_files
        unique_files = {}
        for file_info in all_files:
            if file_info['path'] not in unique_files:
                unique_files[file_info['path']] = file_info

        unique_files_list = list(unique_files.values())

        if not unique_files_list:
            console.print("[yellow]⚠️  No files found to clean up[/yellow]")
            return

        # Select files to delete
        files_to_delete = self.select_files_to_delete(unique_files_list)

        if not files_to_delete:
            console.print("[yellow]🛑 No files selected for deletion[/yellow]")
            return

        # Confirm deletion
        console.print(f"\n[red]🗑️  About to permanently delete {len(files_to_delete)} files:[/red]")
        for file_path in files_to_delete:
            console.print(f"  [red]• {file_path}[/red]")

        if not Confirm.ask("Are you sure you want to proceed?"):
            console.print("[yellow]🛑 Deletion cancelled[/yellow]")
            return

        # Create backup
        backup_dir = self.backup_files(files_to_delete)
        console.print(f"[green]✅ Backup created: {backup_dir}[/green]")

        # Choose removal method
        method = Prompt.ask(
            "Removal method",
            choices=["filter-branch", "bfg"],
            default="filter-branch"
        )

        # Remove files
        success = self.remove_files_from_git(files_to_delete, method)

        if success:
            console.print("[green]✅ Cleanup completed successfully![/green]")
            console.print(f"[dim]Backup location: {backup_dir}[/dim]")
        else:
            console.print("[red]❌ Cleanup failed[/red]")

def main():
    parser = argparse.ArgumentParser(description="Git Cleanup Tool")
    parser.add_argument("--repo", help="Repository path (default: current directory)")
    parser.add_argument("--days", type=int, default=7, help="Days to look back for recent files")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum file size in MB")
    parser.add_argument("--method", choices=["filter-branch", "bfg"], default="filter-branch",
                       help="Method to remove files from history")
    parser.add_argument("--files", nargs="+", help="Specific files to remove")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    tool = GitCleanupTool(args.repo)

    if args.interactive:
        tool.interactive_cleanup()
    elif args.files:
        # Direct file removal
        console.print(f"[red]🗑️  Removing {len(args.files)} files from git history...[/red]")

        if not tool.check_git_repo():
            console.print("[red]❌ Not a git repository[/red]")
            return

        # Create backup
        backup_dir = tool.backup_files(args.files)
        console.print(f"[green]✅ Backup created: {backup_dir}[/green]")

        # Remove files
        success = tool.remove_files_from_git(args.files, args.method)

        if success:
            console.print("[green]✅ Files removed successfully![/green]")
        else:
            console.print("[red]❌ Failed to remove files[/red]")
    else:
        # Show help
        console.print(Panel.fit(
            "[bold blue]Git Cleanup Tool[/bold blue]\n"
            "[dim]Usage: python git_cleanup.py --interactive[/dim]\n"
            "[dim]Or: python git_cleanup.py --files file1 file2[/dim]",
            border_style="blue"
        ))

if __name__ == "__main__":
    main()
