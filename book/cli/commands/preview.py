"""
Preview command implementation for MLSysBook CLI.

Handles starting development servers with live reload for interactive development.
"""

import subprocess
import signal
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console

console = Console()


class PreviewCommand:
    """Handles preview operations for the MLSysBook."""

    def __init__(self, config_manager, chapter_discovery):
        """Initialize preview command.

        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    def preview_full(self, format_type: str = "html") -> bool:
        """Start full preview server for the entire book.

        Args:
            format_type: Format to preview ('html' only supported)

        Returns:
            True if server started successfully, False otherwise
        """
        if format_type != "html":
            console.print(f"[yellow]⚠️ Preview only supports HTML format, not {format_type}[/yellow]")
            return False

        console.print("[blue]🌐 Starting full book preview server...[/blue]")

        # Setup config
        config_name = self.config_manager.setup_symlink(format_type)
        console.print(f"[blue]🔗 Using {config_name}[/blue]")
        console.print("[dim]🛑 Press Ctrl+C to stop the server[/dim]")

        try:
            # Start Quarto preview server
            preview_cmd = ["quarto", "preview"]

            console.print(f"[blue]💻 Command: {' '.join(preview_cmd)}[/blue]")

            # Run preview server (this will block until Ctrl+C)
            result = subprocess.run(
                preview_cmd,
                cwd=self.config_manager.book_dir
            )

            return result.returncode == 0

        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Preview server stopped[/yellow]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Preview server error: {e}[/red]")
            return False

    def preview_chapter(self, chapter_name: str) -> bool:
        """Start preview server for a specific chapter.

        Args:
            chapter_name: Name of the chapter to preview

        Returns:
            True if server started successfully, False otherwise
        """
        # Find the chapter file
        chapter_file = self.chapter_discovery.find_chapter_file(chapter_name)
        if not chapter_file:
            console.print(f"[red]❌ No chapter found matching '{chapter_name}'[/red]")
            console.print("[yellow]💡 Available chapters:[/yellow]")
            self.chapter_discovery.show_chapters()
            return False

        target_path = str(chapter_file.relative_to(self.config_manager.book_dir))
        chapter_display_name = str(chapter_file.relative_to(self.config_manager.book_dir / "contents")).replace(".qmd", "")

        console.print(f"[blue]🌐 Starting preview for[/blue] [bold]{chapter_display_name}[/bold]")

        # Setup HTML config for preview
        self.config_manager.setup_symlink("html")

        console.print("[dim]🛑 Press Ctrl+C to stop the server[/dim]")

        try:
            # Start Quarto preview for specific file
            preview_cmd = ["quarto", "preview", target_path]

            console.print(f"[blue]💻 Command: {' '.join(preview_cmd)}[/blue]")

            # Run preview server (this will block until Ctrl+C)
            result = subprocess.run(
                preview_cmd,
                cwd=self.config_manager.book_dir
            )

            return result.returncode == 0

        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Preview server stopped[/yellow]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Preview server error: {e}[/red]")
            return False
