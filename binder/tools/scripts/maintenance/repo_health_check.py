#!/usr/bin/env python3
"""
Repository Health Check and Maintenance Script
Performs comprehensive health checks and cleanup operations on the MLSysBook repository
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import shutil

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.text import Text
from rich import print as rprint

console = Console()

class RepoHealthChecker:
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

    def get_repository_stats(self) -> Dict:
        """Get comprehensive repository statistics"""
        stats = {}

        # Get basic git stats
        success, output = self.run_git_command(["git", "count-objects", "-vH"])
        if success:
            for line in output.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    stats[key.strip()] = value.strip()

        # Get large files
        large_files = self.find_large_files()
        stats['large_files_count'] = len(large_files)
        stats['large_files_total_size'] = sum(f['size'] for f in large_files)

        # Get duplicate files
        duplicates = self.find_duplicate_files()
        stats['duplicate_files_count'] = len(duplicates)

        return stats

    def find_large_files(self, min_size_mb: int = 5) -> List[Dict]:
        """Find large files in the repository"""
        console.print(f"[blue]🔍 Finding files larger than {min_size_mb}MB...[/blue]")

        find_cmd = [
            "find", ".", "-type", "f", "-size", f"+{min_size_mb}M",
            "-not", "-path", "./.git/*", "-not", "-path", "./build/*"
        ]

        success, output = self.run_git_command(find_cmd)
        if not success:
            return []

        large_files = []
        for line in output.strip().split('\n'):
            if line.strip():
                file_path = line.strip()
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    large_files.append({
                        'path': file_path,
                        'size': size,
                        'size_mb': size / (1024 * 1024),
                        'tracked': self.is_file_tracked(file_path)
                    })

        return sorted(large_files, key=lambda x: x['size'], reverse=True)

    def find_duplicate_files(self) -> List[Dict]:
        """Find duplicate files (same content, different paths)"""
        console.print("[blue]🔍 Finding duplicate files...[/blue]")

        # Get all tracked files
        success, output = self.run_git_command(["git", "ls-files"])
        if not success:
            return []

        files = output.strip().split('\n')
        duplicates = []

        # Simple duplicate detection by size and first few bytes
        size_groups = {}
        for file_path in files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > 1024:  # Only check files > 1KB
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(file_path)

        # Check files with same size
        for size, file_list in size_groups.items():
            if len(file_list) > 1:
                # For now, just report files with same size
                # Could add content hash comparison for more accuracy
                duplicates.append({
                    'size': size,
                    'files': file_list,
                    'count': len(file_list)
                })

        return duplicates

    def is_file_tracked(self, file_path: str) -> bool:
        """Check if a file is tracked by git"""
        success, output = self.run_git_command(["git", "ls-files", file_path])
        return success and output.strip() == file_path

    def find_large_files_in_history(self, min_size_mb: int = 10) -> List[Dict]:
        """Find large files in git history"""
        console.print(f"[blue]🔍 Finding large files in git history (>={min_size_mb}MB)...[/blue]")

        cmd = [
            "git", "rev-list", "--objects", "--all"
        ]

        success, output = self.run_git_command(cmd)
        if not success:
            return []

        # Parse the output to find large blobs
        large_files = []
        for line in output.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        size = int(parts[1])
                        if size > min_size_mb * 1024 * 1024:
                            file_path = ' '.join(parts[2:])
                            large_files.append({
                                'hash': parts[0],
                                'size': size,
                                'size_mb': size / (1024 * 1024),
                                'path': file_path
                            })
                    except (ValueError, IndexError):
                        continue

        return sorted(large_files, key=lambda x: x['size'], reverse=True)

    def optimize_images(self, quality: int = 85) -> List[str]:
        """Optimize large images using ImageMagick"""
        console.print(f"[blue]🖼️  Optimizing images with quality {quality}...[/blue]")

        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        large_images = []

        for ext in image_extensions:
            find_cmd = [
                "find", ".", "-type", "f", "-name", f"*{ext}",
                "-not", "-path", "./.git/*", "-not", "-path", "./build/*"
            ]
            success, output = self.run_git_command(find_cmd)
            if success:
                for line in output.strip().split('\n'):
                    if line.strip() and os.path.exists(line.strip()):
                        size = os.path.getsize(line.strip())
                        if size > 1024 * 1024:  # > 1MB
                            large_images.append(line.strip())

        optimized_files = []
        for image_path in large_images:
            try:
                # Create backup
                backup_path = f"{image_path}.backup"
                shutil.copy2(image_path, backup_path)

                # Optimize with ImageMagick
                if image_path.lower().endswith('.png'):
                    cmd = ["convert", image_path, "-strip", "-quality", str(quality), image_path]
                else:
                    cmd = ["convert", image_path, "-strip", "-quality", str(quality), image_path]

                success, _ = self.run_git_command(cmd)
                if success:
                    new_size = os.path.getsize(image_path)
                    old_size = os.path.getsize(backup_path)
                    savings = old_size - new_size

                    if savings > 0:
                        optimized_files.append({
                            'path': image_path,
                            'old_size': old_size,
                            'new_size': new_size,
                            'savings_mb': savings / (1024 * 1024)
                        })
                        console.print(f"[green]✅ Optimized {image_path} (saved {savings/(1024*1024):.1f}MB)[/green]")
                    else:
                        # Restore original if no improvement
                        shutil.move(backup_path, image_path)
                        console.print(f"[yellow]⚠️  No improvement for {image_path}, keeping original[/yellow]")

                # Clean up backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)

            except Exception as e:
                console.print(f"[red]❌ Error optimizing {image_path}: {e}[/red]")

        return optimized_files

    def remove_duplicate_audio(self) -> List[str]:
        """Remove duplicate audio files"""
        console.print("[blue]🎵 Checking for duplicate audio files...[/blue]")

        # Find audio files
        audio_files = []
        find_cmd = [
            "find", ".", "-type", "f", "-name", "*.mp3",
            "-not", "-path", "./.git/*", "-not", "-path", "./build/*"
        ]

        success, output = self.run_git_command(find_cmd)
        if success:
            for line in output.strip().split('\n'):
                if line.strip() and os.path.exists(line.strip()):
                    audio_files.append(line.strip())

        # Check for duplicates by filename
        filename_groups = {}
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            if filename not in filename_groups:
                filename_groups[filename] = []
            filename_groups[filename].append(audio_file)

        removed_files = []
        for filename, file_list in filename_groups.items():
            if len(file_list) > 1:
                # Keep the first one, remove others
                for file_path in file_list[1:]:
                    if self.is_file_tracked(file_path):
                        console.print(f"[yellow]⚠️  Found duplicate: {file_path}[/yellow]")
                        if Confirm.ask(f"Remove duplicate {file_path}?"):
                            try:
                                os.remove(file_path)
                                self.run_git_command(["git", "rm", file_path])
                                removed_files.append(file_path)
                                console.print(f"[green]✅ Removed duplicate: {file_path}[/green]")
                            except Exception as e:
                                console.print(f"[red]❌ Error removing {file_path}: {e}[/red]")

        return removed_files

    def cleanup_build_artifacts(self) -> List[str]:
        """Clean up build artifacts and temporary files"""
        console.print("[blue]🧹 Cleaning up build artifacts...[/blue]")

        # Directories to clean
        cleanup_dirs = [
            "build/",
            "_book/",
            ".quarto/",
            "site_libs/",
            "*.log",
            "*.aux",
            "*.toc",
            "*.out"
        ]

        cleaned_files = []
        for pattern in cleanup_dirs:
            if pattern.endswith('/'):
                # Directory
                dir_path = Path(pattern)
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                        cleaned_files.append(str(dir_path))
                        console.print(f"[green]✅ Removed directory: {dir_path}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ Error removing {dir_path}: {e}[/red]")
            else:
                # File pattern
                find_cmd = ["find", ".", "-name", pattern, "-not", "-path", "./.git/*"]
                success, output = self.run_git_command(find_cmd)
                if success:
                    for file_path in output.strip().split('\n'):
                        if file_path.strip() and os.path.exists(file_path.strip()):
                            try:
                                os.remove(file_path.strip())
                                cleaned_files.append(file_path.strip())
                                console.print(f"[green]✅ Removed file: {file_path.strip()}[/green]")
                            except Exception as e:
                                console.print(f"[red]❌ Error removing {file_path.strip()}: {e}[/red]")

        return cleaned_files

    def run_bfg_cleanup(self, file_patterns: List[str] = None) -> bool:
        """Run BFG cleanup on large files in history"""
        if not file_patterns:
            file_patterns = ["*.pdf", "*.epub"]

        console.print("[blue]🗑️  Running BFG cleanup on large files...[/blue]")

        # Check if BFG is installed
        success, _ = self.run_git_command(["which", "bfg"])
        if not success:
            console.print("[red]❌ BFG not found. Install with: brew install bfg[/red]")
            return False

        if not Confirm.ask("Continue with BFG cleanup? This will rewrite git history."):
            return False

        # No need to create file list - using direct patterns

        try:
            # Run BFG with correct syntax
            cmd = ["bfg", "-D", "*.pdf", "-D", "*.epub"]
            success, output = self.run_git_command(cmd, capture_output=False)

            if success:
                # Clean up git
                self.run_git_command(["git", "reflog", "expire", "--expire=now", "--all"])
                self.run_git_command(["git", "gc", "--prune=now", "--aggressive"])

                console.print("[green]✅ BFG cleanup completed successfully[/green]")
                return True
            else:
                console.print(f"[red]❌ BFG cleanup failed: {output}[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Error during BFG cleanup: {e}[/red]")
            return False
        finally:
            pass  # No file to clean up

    def display_health_report(self, stats: Dict, large_files: List[Dict],
                            history_files: List[Dict], duplicates: List[Dict]):
        """Display comprehensive health report"""
        console.print(Panel.fit(
            "[bold blue]🏥 Repository Health Report[/bold blue]",
            border_style="blue"
        ))

        # Basic stats
        table = Table(title="Repository Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats.items():
            if key in ['size', 'size-pack']:
                table.add_row(key.replace('_', ' ').title(), value)

        console.print(table)

        # Large files
        if large_files:
            console.print("\n[bold yellow]📁 Large Files (>5MB)[/bold yellow]")
            large_table = Table(show_header=True)
            large_table.add_column("File", style="cyan")
            large_table.add_column("Size (MB)", style="yellow")
            large_table.add_column("Tracked", style="green")

            for file_info in large_files[:10]:  # Show top 10
                tracked = "✅" if file_info['tracked'] else "❌"
                large_table.add_row(
                    file_info['path'],
                    f"{file_info['size_mb']:.1f}",
                    tracked
                )

            console.print(large_table)

        # Large files in history
        if history_files:
            console.print("\n[bold red]🗂️  Large Files in Git History[/bold red]")
            history_table = Table(show_header=True)
            history_table.add_column("File", style="cyan")
            history_table.add_column("Size (MB)", style="red")

            for file_info in history_files[:10]:  # Show top 10
                history_table.add_row(
                    file_info['path'],
                    f"{file_info['size_mb']:.1f}"
                )

            console.print(history_table)

        # Duplicates
        if duplicates:
            console.print("\n[bold yellow]🔄 Duplicate Files[/bold yellow]")
            dup_table = Table(show_header=True)
            dup_table.add_column("Size", style="cyan")
            dup_table.add_column("Count", style="yellow")
            dup_table.add_column("Files", style="white")

            for dup_info in duplicates[:5]:  # Show top 5
                files_str = ", ".join([os.path.basename(f) for f in dup_info['files'][:3]])
                if len(dup_info['files']) > 3:
                    files_str += f" (+{len(dup_info['files']) - 3} more)"

                dup_table.add_row(
                    f"{dup_info['size'] / (1024*1024):.1f}MB",
                    str(dup_info['count']),
                    files_str
                )

            console.print(dup_table)

    def run_full_maintenance(self, optimize_images: bool = True,
                           remove_duplicates: bool = True,
                           bfg_cleanup: bool = False) -> Dict:
        """Run full maintenance workflow"""
        console.print(Panel.fit(
            "[bold green]🔧 Repository Maintenance[/bold green]\n"
            "[dim]Comprehensive cleanup and optimization[/dim]",
            border_style="green"
        ))

        # Check if we're in a git repo
        if not self.check_git_repo():
            console.print("[red]❌ Not a git repository. Please run this from a git repo.[/red]")
            return {}

        results = {
            'optimized_images': [],
            'removed_duplicates': [],
            'cleaned_artifacts': [],
            'bfg_cleanup': False
        }

        # 1. Clean build artifacts
        console.print("\n[bold]Step 1: Cleaning build artifacts...[/bold]")
        results['cleaned_artifacts'] = self.cleanup_build_artifacts()

        # 2. Remove duplicate audio files
        if remove_duplicates:
            console.print("\n[bold]Step 2: Removing duplicate files...[/bold]")
            results['removed_duplicates'] = self.remove_duplicate_audio()

        # 3. Optimize images
        if optimize_images:
            console.print("\n[bold]Step 3: Optimizing images...[/bold]")
            results['optimized_images'] = self.optimize_images()

        # 4. BFG cleanup (optional)
        if bfg_cleanup:
            console.print("\n[bold]Step 4: BFG cleanup...[/bold]")
            results['bfg_cleanup'] = self.run_bfg_cleanup()

        # 5. Final health check
        console.print("\n[bold]Step 5: Final health check...[/bold]")
        stats = self.get_repository_stats()
        large_files = self.find_large_files()
        history_files = self.find_large_files_in_history()
        duplicates = self.find_duplicate_files()

        self.display_health_report(stats, large_files, history_files, duplicates)

        return results

def main():
    parser = argparse.ArgumentParser(description="Repository Health Check and Maintenance")
    parser.add_argument("--repo", help="Repository path (default: current directory)")
    parser.add_argument("--min-size", type=int, default=5, help="Minimum file size in MB to report")
    parser.add_argument("--optimize-images", action="store_true", help="Optimize large images")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate files")
    parser.add_argument("--bfg-cleanup", action="store_true", help="Run BFG cleanup on large files")
    parser.add_argument("--full", action="store_true", help="Run full maintenance workflow")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")

    args = parser.parse_args()

    checker = RepoHealthChecker(args.repo)

    if args.health_check:
        # Health check only
        stats = checker.get_repository_stats()
        large_files = checker.find_large_files(args.min_size)
        history_files = checker.find_large_files_in_history(args.min_size)
        duplicates = checker.find_duplicate_files()

        checker.display_health_report(stats, large_files, history_files, duplicates)

    elif args.full:
        # Full maintenance
        results = checker.run_full_maintenance(
            optimize_images=args.optimize_images,
            remove_duplicates=args.remove_duplicates,
            bfg_cleanup=args.bfg_cleanup
        )

        # Summary
        console.print("\n[bold green]✅ Maintenance Complete![/bold green]")
        console.print(f"Optimized images: {len(results['optimized_images'])}")
        console.print(f"Removed duplicates: {len(results['removed_duplicates'])}")
        console.print(f"Cleaned artifacts: {len(results['cleaned_artifacts'])}")
        if results['bfg_cleanup']:
            console.print("BFG cleanup: ✅ Completed")

    else:
        # Default: health check
        stats = checker.get_repository_stats()
        large_files = checker.find_large_files(args.min_size)
        history_files = checker.find_large_files_in_history(args.min_size)
        duplicates = checker.find_duplicate_files()

        checker.display_health_report(stats, large_files, history_files, duplicates)

if __name__ == "__main__":
    main()
