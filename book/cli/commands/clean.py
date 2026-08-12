"""
Clean command implementation for MLSysBook CLI.

Handles cleaning build artifacts and restoring configurations.
"""

import shutil
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.panel import Panel
from rich.table import Table

from cli.core.artifacts import clean_build_artifacts

console = Console()


class CleanCommand:
    """Handles cleaning operations for the MLSysBook."""

    def __init__(self, config_manager, chapter_discovery):
        """Initialize clean command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def clean_all(self) -> bool:
        """Clean all build artifacts and restore configs.

        Returns:
            True if cleaning succeeded, False otherwise
        """
        console.print("[bold blue]🧹 MLSysBook Cleanup[/bold blue]")
        console.print("[dim]Cleaning build artifacts and restoring configurations...[/dim]\n")

        success = True

        # Clean build directories
        success &= self._clean_build_directories()

        # Clean temporary files
        success &= self._clean_temp_files()

        # Restore configurations
        success &= self._restore_configs()

        # Show current status
        self._show_cleanup_status()

        if success:
            console.print("\n[green]✅ Cleanup completed successfully![/green]")
        else:
            console.print("\n[yellow]⚠️ Cleanup completed with some issues[/yellow]")

        return success

    def _clean_build_directories(self) -> bool:
        """Clean build output directories."""
        console.print("[blue]📁 Cleaning build directories...[/blue]")

        formats = ["html", "pdf", "epub"]
        cleaned_dirs = []

        for format_type in formats:
            try:
                output_dir = self.config_manager.get_output_dir(format_type)
                if output_dir.exists():
                    # Count files before deletion
                    files = list(output_dir.rglob("*"))
                    file_count = len([f for f in files if f.is_file()])

                    # Remove the directory
                    shutil.rmtree(output_dir)
                    cleaned_dirs.append(f"{format_type.upper()}: {file_count} files")
                    console.print(f"  ✅ Cleaned {format_type.upper()} build ({file_count} files)")
                else:
                    console.print(f"  📁 {format_type.upper()} build directory not found (already clean)")

            except Exception as e:
                console.print(f"  ❌ Error cleaning {format_type.upper()}: {e}")
                return False

        return True

    def _clean_temp_files(self) -> bool:
        """Clean temporary files and caches."""
        console.print("[blue]🗑️ Cleaning temporary files...[/blue]")

        temp_patterns = [
            "**/.quarto",
            "**/.*_cache",
            "**/*_files",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.log"
        ]

        cleaned_count = 0

        for pattern in temp_patterns:
            try:
                for temp_file in self.config_manager.book_dir.glob(pattern):
                    if temp_file.exists():
                        if temp_file.is_file():
                            temp_file.unlink()
                            cleaned_count += 1
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                            cleaned_count += 1
            except Exception as e:
                console.print(f"  ⚠️ Warning cleaning {pattern}: {e}")

        console.print(f"  ✅ Cleaned {cleaned_count} temporary files/directories")
        return True

    def _restore_configs(self) -> bool:
        """Restore configuration files to clean state."""
        console.print("[blue]⚙️ Restoring configurations...[/blue]")

        try:
            # Remove any active symlink
            if self.config_manager.active_config.exists():
                if self.config_manager.active_config.is_symlink():
                    target = self.config_manager.active_config.readlink()
                    self.config_manager.active_config.unlink()
                    console.print(f"  ✅ Removed symlink to {target}")
                else:
                    console.print("  📄 Active config is a regular file (not removing)")
            else:
                console.print("  📁 No active config found")

            return True

        except Exception as e:
            console.print(f"  ❌ Error restoring configs: {e}")
            return False

    def _show_cleanup_status(self) -> None:
        """Show current status after cleanup."""
        console.print("\n[blue]📊 Post-cleanup status:[/blue]")

        # Show symlink status
        self.config_manager.show_symlink_status()

        # Show build directory status
        for format_type in ["html", "pdf", "epub"]:
            output_dir = self.config_manager.get_output_dir(format_type)
            if output_dir.exists():
                files = list(output_dir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                console.print(f"  📁 {format_type.upper()}: {file_count} files remaining")
            else:
                console.print(f"  📁 {format_type.upper()}: Clean (no build directory)")

    def clean_format(self, format_type: str) -> bool:
        """Clean artifacts for a specific format.

        Args:
            format_type: Format to clean ('html', 'pdf', 'epub')

        Returns:
            True if cleaning succeeded, False otherwise
        """
        console.print(f"[blue]🧹 Cleaning {format_type.upper()} artifacts...[/blue]")

        try:
            output_dir = self.config_manager.get_output_dir(format_type)
            if output_dir.exists():
                files = list(output_dir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])

                shutil.rmtree(output_dir)
                console.print(f"✅ Cleaned {file_count} {format_type.upper()} files")
            else:
                console.print(f"📁 {format_type.upper()} directory not found (already clean)")

            return True

        except Exception as e:
            console.print(f"❌ Error cleaning {format_type.upper()}: {e}")
            return False

    def clean_artifacts(self) -> bool:
        """Clean build artifacts using Binder-native cleanup primitives.

        This is the same cleanup that runs as a pre-commit hook.
        """
        clean_build_artifacts(self.config_manager.book_dir, console=console)
        return True

    def print_help(self) -> None:
        """Print the `binder clean` command reference."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Target", style="cyan", width=14)
        table.add_column("Description", style="white", width=58)
        table.add_row("(none)", "Clean HTML/PDF/EPUB build directories, temp files, and active config symlink")
        table.add_row("html", "Clean the active HTML output directory")
        table.add_row("pdf", "Clean the active PDF output directory")
        table.add_row("epub", "Clean the active EPUB output directory")
        table.add_row("artifacts", "Clean generated build artifacts using the same native primitive used by pre-commit")
        console.print(Panel(table, title=_rich_escape("binder clean [target]"), border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder clean[/cyan]            [dim]# full local cleanup[/dim]")
        console.print("  [cyan]./binder clean html[/cyan]       [dim]# clean HTML output only[/dim]")
        console.print("  [cyan]./binder clean artifacts[/cyan]  [dim]# generated-artifact cleanup[/dim]")
        console.print()
