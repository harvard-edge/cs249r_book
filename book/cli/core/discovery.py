"""
File and chapter discovery for MLSysBook CLI.

Handles finding chapter files, validating paths, and managing file operations.
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console

console = Console()


class ChapterDiscovery:
    """Discovers and manages chapter files in the MLSysBook project."""

    def __init__(self, book_dir: Path):
        """Initialize chapter discovery.

        Args:
            book_dir: Path to the book directory (usually 'quarto')
        """
        self.book_dir = Path(book_dir)
        self.contents_dir = self.book_dir / "contents"

    def find_chapter_file(self, chapter_name: str) -> Optional[Path]:
        """Find a chapter file by name or partial match.

        Args:
            chapter_name: Chapter name to search for

        Returns:
            Path to the chapter file if found, None otherwise
        """
        if not self.contents_dir.exists():
            console.print(f"[red]❌ Contents directory not found: {self.contents_dir}[/red]")
            return None

        # Try exact match first
        exact_matches = list(self.contents_dir.rglob(f"{chapter_name}.qmd"))
        if exact_matches:
            return exact_matches[0]

        # Try partial matches
        pattern = f"*{chapter_name}*.qmd"
        partial_matches = list(self.contents_dir.rglob(pattern))

        if not partial_matches:
            return None

        if len(partial_matches) == 1:
            return partial_matches[0]

        # Multiple matches - try to find the best one
        # Prefer files where the chapter name is part of the directory or filename
        for match in partial_matches:
            if chapter_name in match.stem or chapter_name in match.parent.name:
                return match

        # Return the first match as fallback
        return partial_matches[0]

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        """Get all chapter files with metadata.

        Returns:
            List of dictionaries containing chapter information
        """
        chapters = []

        if not self.contents_dir.exists():
            return chapters

        for qmd_file in self.contents_dir.rglob("*.qmd"):
            # Skip certain files
            if qmd_file.name in ["index.qmd", "404.qmd"]:
                continue

            # Get relative path from contents directory
            rel_path = qmd_file.relative_to(self.contents_dir)

            # Extract chapter info
            chapter_info = {
                "name": qmd_file.stem,
                "path": qmd_file,
                "relative_path": rel_path,
                "directory": qmd_file.parent.name,
                "size": qmd_file.stat().st_size if qmd_file.exists() else 0
            }

            chapters.append(chapter_info)

        # Sort by path for consistent ordering
        chapters.sort(key=lambda x: str(x["relative_path"]))
        return chapters

    def show_chapters(self) -> None:
        """Display available chapters in a formatted table."""
        from rich.table import Table

        chapters = self.get_all_chapters()

        if not chapters:
            console.print("[yellow]No chapters found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Chapter", style="green", width=25)
        table.add_column("Directory", style="cyan", width=20)
        table.add_column("Size", style="dim", width=10)

        for chapter in chapters:
            size_kb = chapter["size"] / 1024 if chapter["size"] > 0 else 0
            size_str = f"{size_kb:.1f} KB" if size_kb > 0 else "0 KB"

            table.add_row(
                chapter["name"],
                chapter["directory"],
                size_str
            )

        console.print(table)
        console.print(f"\n[dim]Found {len(chapters)} chapters[/dim]")

    def validate_chapters(self, chapter_names: List[str]) -> List[Path]:
        """Validate a list of chapter names and return their paths.

        Args:
            chapter_names: List of chapter names to validate

        Returns:
            List of valid chapter file paths

        Raises:
            FileNotFoundError: If any chapter is not found
        """
        chapter_files = []

        for chapter_name in chapter_names:
            chapter_file = self.find_chapter_file(chapter_name)
            if not chapter_file:
                available_chapters = [ch["name"] for ch in self.get_all_chapters()]
                console.print(f"[red]❌ Chapter not found: {chapter_name}[/red]")
                console.print("[yellow]💡 Available chapters:[/yellow]")
                for ch in available_chapters[:10]:  # Show first 10
                    console.print(f"  - {ch}")
                if len(available_chapters) > 10:
                    console.print(f"  ... and {len(available_chapters) - 10} more")
                raise FileNotFoundError(f"Chapter not found: {chapter_name}")

            chapter_files.append(chapter_file)

        return chapter_files

    def get_chapter_dependencies(self, chapter_file: Path) -> List[Path]:
        """Get dependencies for a chapter (images, includes, etc.).

        Args:
            chapter_file: Path to the chapter file

        Returns:
            List of dependency file paths
        """
        dependencies = []

        if not chapter_file.exists():
            return dependencies

        try:
            content = chapter_file.read_text(encoding='utf-8')

            # Find image references
            image_pattern = r'!\[.*?\]\((.*?)\)'
            for match in re.finditer(image_pattern, content):
                image_path = match.group(1)
                if not image_path.startswith('http'):
                    # Resolve relative to chapter file
                    full_path = (chapter_file.parent / image_path).resolve()
                    if full_path.exists():
                        dependencies.append(full_path)

            # Find include references
            include_pattern = r'{{< include (.*?) >}}'
            for match in re.finditer(include_pattern, content):
                include_path = match.group(1)
                full_path = (chapter_file.parent / include_path).resolve()
                if full_path.exists():
                    dependencies.append(full_path)

        except Exception as e:
            console.print(f"[yellow]⚠️  Error reading chapter dependencies: {e}[/yellow]")

        return dependencies
