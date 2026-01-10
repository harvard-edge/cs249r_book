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

        # Configuration file paths (combined configs)
        self.html_config = self.book_dir / "config" / "_quarto-html.yml"
        self.pdf_config = self.book_dir / "config" / "_quarto-pdf.yml"
        self.epub_config = self.book_dir / "config" / "_quarto-epub.yml"

        # Volume-specific configuration file paths
        self.html_vol1_config = self.book_dir / "config" / "_quarto-html-vol1.yml"
        self.html_vol2_config = self.book_dir / "config" / "_quarto-html-vol2.yml"
        self.pdf_vol1_config = self.book_dir / "config" / "_quarto-pdf-vol1.yml"
        self.pdf_vol2_config = self.book_dir / "config" / "_quarto-pdf-vol2.yml"
        self.epub_vol1_config = self.book_dir / "config" / "_quarto-epub-vol1.yml"
        self.epub_vol2_config = self.book_dir / "config" / "_quarto-epub-vol2.yml"

        self.active_config = self.book_dir / "_quarto.yml"

    def get_config_file(self, format_type: str, volume: Optional[str] = None) -> Path:
        """Get the configuration file for a specific format and optional volume.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume ('vol1', 'vol2') for volume-specific builds

        Returns:
            Path to the configuration file

        Raises:
            ValueError: If format_type is not supported
        """
        # Volume-specific config map
        if volume:
            volume_config_map = {
                ("html", "vol1"): self.html_vol1_config,
                ("html", "vol2"): self.html_vol2_config,
                ("pdf", "vol1"): self.pdf_vol1_config,
                ("pdf", "vol2"): self.pdf_vol2_config,
                ("epub", "vol1"): self.epub_vol1_config,
                ("epub", "vol2"): self.epub_vol2_config,
            }
            key = (format_type, volume)
            if key in volume_config_map:
                config_file = volume_config_map[key]
                if config_file.exists():
                    return config_file
                else:
                    console.print(f"[yellow]⚠️ Volume config not found: {config_file}, falling back to combined config[/yellow]")

        # Combined config map (fallback)
        config_map = {
            "html": self.html_config,
            "pdf": self.pdf_config,
            "epub": self.epub_config
        }

        if format_type not in config_map:
            raise ValueError(f"Unsupported format type: {format_type}")

        return config_map[format_type]

    def setup_symlink(self, format_type: str, volume: Optional[str] = None) -> str:
        """Setup _quarto.yml symlink for the specified format and optional volume.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume ('vol1', 'vol2') for volume-specific builds

        Returns:
            Name of the config file that was linked

        Raises:
            ValueError: If format_type is not supported
        """
        config_file = self.get_config_file(format_type, volume)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Remove existing symlink/file
        if self.active_config.exists() or self.active_config.is_symlink():
            self.active_config.unlink()

        # Create new symlink
        relative_path = config_file.relative_to(self.book_dir)
        self.active_config.symlink_to(relative_path)

        return config_file.name

    def get_output_dir(self, format_type: str, volume: Optional[str] = None) -> Path:
        """Get the output directory from Quarto configuration.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume ('vol1', 'vol2') for volume-specific builds

        Returns:
            Path to the output directory
        """
        try:
            config_file = self.get_config_file(format_type, volume)

            if not config_file.exists():
                console.print(f"[yellow]⚠️  Config file not found: {config_file}[/yellow]")
                # Fallback to default
                suffix = f"-{volume}" if volume else ""
                return self.book_dir / f"_build/{format_type}{suffix}"

            # Read and parse the YAML config
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Extract output directory from project.output-dir
            if config and 'project' in config and 'output-dir' in config['project']:
                output_path = config['project']['output-dir']
                return self.book_dir / output_path
            else:
                # Fallback to default
                suffix = f"-{volume}" if volume else ""
                return self.book_dir / f"_build/{format_type}{suffix}"

        except Exception as e:
            console.print(f"[yellow]⚠️  Error reading config: {e}[/yellow]")
            suffix = f"-{volume}" if volume else ""
            return self.book_dir / f"_build/{format_type}{suffix}"

    def read_config(self, format_type: str, volume: Optional[str] = None) -> Dict[str, Any]:
        """Read and parse a configuration file.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume ('vol1', 'vol2') for volume-specific builds

        Returns:
            Parsed configuration as dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_file = self.get_config_file(format_type, volume)

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
