"""
Configuration management for MLSysBook CLI.

Handles Quarto configuration files, symlinks, and format-specific settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()


class ConfigManager:
    """Manages Quarto configuration files and format switching."""

    def __init__(self, root_dir: Path):
        """Initialize configuration manager.

        Args:
            root_dir: Root directory of the MLSysBook project
        """
        self.root_dir = Path(root_dir)

        # Determine book directory
        if (self.root_dir / "book" / "quarto").exists():
            # New structure: book/quarto/
            self.book_dir = self.root_dir / "book" / "quarto"
        elif (self.root_dir / "quarto").exists():
            # Old structure or running from book/: quarto/
            self.book_dir = self.root_dir / "quarto"
        else:
            # We're in quarto directory
            self.book_dir = self.root_dir

        # Configuration file paths
        self.html_config = self.book_dir / "config" / "_quarto-html.yml"
        self.pdf_config = self.book_dir / "config" / "_quarto-pdf.yml"
        self.epub_config = self.book_dir / "config" / "_quarto-epub.yml"
        self.active_config = self.book_dir / "_quarto.yml"

    def get_config_file(self, format_type: str) -> Path:
        """Get the configuration file for a specific format.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')

        Returns:
            Path to the configuration file

        Raises:
            ValueError: If format_type is not supported
        """
        config_map = {
            "html": self.html_config,
            "pdf": self.pdf_config,
            "epub": self.epub_config
        }

        if format_type not in config_map:
            raise ValueError(f"Unsupported format type: {format_type}")

        return config_map[format_type]

    def setup_symlink(self, format_type: str) -> str:
        """Setup _quarto.yml symlink for the specified format.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')

        Returns:
            Name of the config file that was linked

        Raises:
            ValueError: If format_type is not supported
        """
        config_file = self.get_config_file(format_type)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Remove existing symlink/file
        if self.active_config.exists() or self.active_config.is_symlink():
            self.active_config.unlink()

        # Create new symlink
        relative_path = config_file.relative_to(self.book_dir)
        self.active_config.symlink_to(relative_path)

        return config_file.name

    def get_output_dir(self, format_type: str) -> Path:
        """Get the output directory from Quarto configuration.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')

        Returns:
            Path to the output directory
        """
        try:
            config_file = self.get_config_file(format_type)

            if not config_file.exists():
                console.print(f"[yellow]⚠️  Config file not found: {config_file}[/yellow]")
                # Fallback to default
                return self.book_dir / f"_build/{format_type}"

            # Read and parse the YAML config
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Extract output directory from project.output-dir
            if config and 'project' in config and 'output-dir' in config['project']:
                output_path = config['project']['output-dir']
                return self.book_dir / output_path
            else:
                # Fallback to default
                return self.book_dir / f"_build/{format_type}"

        except Exception as e:
            console.print(f"[yellow]⚠️  Error reading config: {e}[/yellow]")
            return self.book_dir / f"_build/{format_type}"

    def read_config(self, format_type: str) -> Dict[str, Any]:
        """Read and parse a configuration file.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')

        Returns:
            Parsed configuration as dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_file = self.get_config_file(format_type)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def show_symlink_status(self) -> None:
        """Display current symlink status."""
        if self.active_config.is_symlink():
            target = self.active_config.readlink()
            console.print(f"[dim]  🔗 Active config: {target}[/dim]")
        elif self.active_config.exists():
            console.print("[dim]  📄 Active config: _quarto.yml (regular file)[/dim]")
        else:
            console.print("[dim]  ❌ No active config found[/dim]")
