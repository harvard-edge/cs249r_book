"""
Preview command implementation for MLSysBook CLI.

Handles starting development servers with live reload for interactive development.
"""

import subprocess
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
            True if the server exited cleanly or was stopped with Ctrl+C.
        """
        if format_type != "html":
            console.print(f"[red]❌ Preview supports HTML only, not {format_type}. "
                          f"Build other formats with: ./binder/binder build {format_type}[/red]")
            return False

        console.print("[blue]🌐 Starting full book preview server...[/blue]")
        config_name = self.config_manager.activate_config(format_type)
        console.print(f"[blue]🔗 Using {config_name}[/blue]")
        return self._serve(["quarto", "preview"])

    def preview_chapter(self, chapter_name: str) -> bool:
        """Start preview server for a specific chapter.

        Args:
            chapter_name: Name of the chapter to preview

        Returns:
            True if the server exited cleanly or was stopped with Ctrl+C.
        """
        chapter_file = self.chapter_discovery.find_chapter_file(chapter_name)
        if not chapter_file:
            console.print(f"[red]❌ No chapter found matching '{chapter_name}'[/red]")
            console.print("[yellow]💡 Available chapters:[/yellow]")
            self.chapter_discovery.show_chapters()
            return False

        target_path = chapter_file.relative_to(self.config_manager.book_dir).as_posix()
        console.print(f"[blue]🌐 Starting preview for[/blue] [bold]{target_path.removesuffix('.qmd')}[/bold]")
        self.config_manager.activate_config("html")
        return self._serve(["quarto", "preview", target_path])

    def _serve(self, preview_cmd: list) -> bool:
        """Run ``quarto preview`` in the books directory until it exits or Ctrl+C stops it."""
        console.print("[dim]🛑 Press Ctrl+C to stop the server[/dim]")
        console.print(f"[blue]💻 Command: {' '.join(preview_cmd)}[/blue]")
        try:
            result = subprocess.run(preview_cmd, cwd=self.config_manager.book_dir)
        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Preview server stopped[/yellow]")
            return True
        except FileNotFoundError as e:
            console.print(f"[red]❌ Could not start quarto: {e}[/red]")
            console.print("[yellow]Check that Quarto is installed and on PATH: ./binder/binder doctor[/yellow]")
            return False
        if result.returncode != 0:
            console.print(f"[red]❌ {' '.join(preview_cmd)} exited with code {result.returncode} "
                          f"(in {self.config_manager.book_dir}); see its output above.[/red]")
            return False
        return True
