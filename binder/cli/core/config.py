"""
Configuration management for MLSysBook CLI.

Handles Quarto configuration files, the generated _quarto.yml, and format-specific settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from .volume_index import volume_index_source, write_volume_index

console = Console()

ACTIVE_CONFIG_MARKER = "# binder: generated copy of "


def write_active_config(active_config: Path, source: Path, book_dir: Path) -> None:
    """Write *source* to the project's ``_quarto.yml`` with a provenance header.

    Quarto only reads ``_quarto.yml``, so every build copies the chosen
    configuration there. The first line records the source file so status
    output and post-render scripts can tell which configuration is active.

    Args:
        active_config: Destination path (usually ``book_dir / "_quarto.yml"``).
        source: Source configuration file to copy.
        book_dir: Project books root directory for relative provenance path.
    """
    header = (
        f"{ACTIVE_CONFIG_MARKER}{source.relative_to(book_dir).as_posix()}\n"
        "# Regenerated on every build; edit the source file, not this copy.\n"
    )
    # Earlier binder versions left a symlink here. Writing through it would
    # overwrite the source configuration, so remove it first.
    if active_config.is_symlink():
        active_config.unlink()
    active_config.write_text(header + source.read_text(encoding="utf-8"), encoding="utf-8")


def active_config_source(active_config: Path) -> Optional[str]:
    """Return the source file recorded in a generated ``_quarto.yml``, or None.

    Args:
        active_config: Path to the active ``_quarto.yml`` file.

    Returns:
        Relative path string recorded in the header marker, or None if invalid or missing.
    """
    if not active_config.is_file():
        return None
    with active_config.open(encoding="utf-8") as fh:
        first_line = fh.readline()
    if not first_line.startswith(ACTIVE_CONFIG_MARKER):
        return None
    return first_line[len(ACTIVE_CONFIG_MARKER):].strip()


def get_output_file(output_dir: Path, format_type: str) -> Optional[Path]:
    """Return the primary output file for a build: any .pdf, any .epub, or index.html.

    Used by build (open output) and the parallel runner (success check) so all
    commands use the same rule: PDF = first .pdf in dir, EPUB = first .epub in
    dir, HTML = index.html.

    Args:
        output_dir: Directory where build output was emitted.
        format_type: Output format ("html", "pdf", or "epub").

    Returns:
        Path to primary output file if found, otherwise None.
    """
    if not output_dir.exists():
        return None
    if format_type == "pdf":
        for p in sorted(output_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".pdf":
                return p
        return None
    if format_type == "epub":
        for p in sorted(output_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".epub":
                return p
        return None
    if format_type == "html":
        index = output_dir / "index.html"
        return index if index.exists() else None
    return None


class ConfigManager:
    """Manages Quarto configuration files and format switching."""

    def __init__(self, root_dir: Path):
        """Initialize configuration manager.

        Args:
            root_dir: Root directory of the MLSysBook project
        """
        self.root_dir = Path(root_dir)

        # Determine the book directory, which is the Quarto project root.
        #
        # Book sources live in books/ at the repository root: one directory per
        # volume, plus shared/ for anything cross-volume, plus the config/ and
        # _extensions/ that Quarto needs at its project root. Before 2026-09
        # this lived at books/ with the volumes under contents/,
        # and books/ held a second, drifting copy that fed nothing. See
        # docs/REPO_LAYOUT.md.
        if (self.root_dir / "books" / "config").exists():
            # Running from the repository root
            self.book_dir = self.root_dir / "books"
        elif (self.root_dir / "config").exists() and (self.root_dir / "vol1").exists():
            # Already inside books/
            self.book_dir = self.root_dir
        elif (self.root_dir.parent / "books" / "config").exists():
            # Running from a sibling directory such as binder/
            self.book_dir = self.root_dir.parent / "books"
        else:
            # Fallback
            self.book_dir = self.root_dir

        # Configuration file paths (default to vol1 configs since combined configs don't exist)
        self.html_config = self.book_dir / "config" / "_quarto-html-vol1.yml"
        self.pdf_config = self.book_dir / "config" / "_quarto-pdf-vol1.yml"
        self.epub_config = self.book_dir / "config" / "_quarto-epub-vol1.yml"

        # Volume-specific configuration file paths
        self.html_vol1_config = self.book_dir / "config" / "_quarto-html-vol1.yml"
        self.html_vol2_config = self.book_dir / "config" / "_quarto-html-vol2.yml"
        self.html_vol3_config = self.book_dir / "config" / "_quarto-html-vol3.yml"
        self.html_vol4_config = self.book_dir / "config" / "_quarto-html-vol4.yml"
        self.pdf_vol1_config = self.book_dir / "config" / "_quarto-pdf-vol1.yml"
        self.pdf_vol2_config = self.book_dir / "config" / "_quarto-pdf-vol2.yml"
        self.pdf_vol3_config = self.book_dir / "config" / "_quarto-pdf-vol3.yml"
        self.pdf_vol4_config = self.book_dir / "config" / "_quarto-pdf-vol4.yml"
        self.epub_vol1_config = self.book_dir / "config" / "_quarto-epub-vol1.yml"
        self.epub_vol2_config = self.book_dir / "config" / "_quarto-epub-vol2.yml"
        self.epub_vol3_config = self.book_dir / "config" / "_quarto-epub-vol3.yml"
        self.epub_vol4_config = self.book_dir / "config" / "_quarto-epub-vol4.yml"

        self.active_config = self.book_dir / "_quarto.yml"
        self.active_index = self.book_dir / "index.qmd"

        # Canonical sources shared with the Linux and Windows CI builds.
        self.index_vol1 = self.book_dir / "index-vol1.qmd"
        self.index_vol2 = self.book_dir / "index-vol2.qmd"
        self.index_vol3 = self.book_dir / "index-vol3.qmd"
        self.index_vol4 = volume_index_source(self.book_dir, "vol4", "pdf")
        self.html_index_vol4 = volume_index_source(self.book_dir, "vol4", "html")

    def get_config_file(self, format_type: str, volume: Optional[str] = None) -> Path:
        """Get the configuration file for a specific format and optional volume.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume (e.g. 'vol1', 'vol2', 'vol3', 'vol4', 'vol5')

        Returns:
            Path to the configuration file

        Raises:
            ValueError: If format_type is not supported
        """
        if format_type not in ("html", "pdf", "epub"):
            raise ValueError(f"Unsupported format type: {format_type}")

        if volume:
            config_file = self.book_dir / "config" / f"_quarto-{format_type}-{volume}.yml"
            if config_file.exists():
                return config_file
            console.print(f"[yellow]⚠️ Volume config not found: {config_file}, falling back to default config[/yellow]")

        default_config = self.book_dir / "config" / f"_quarto-{format_type}-vol1.yml"
        if default_config.exists():
            return default_config
        return self.book_dir / "config" / f"_quarto-{format_type}.yml"

    def activate_config(self, format_type: str, volume: Optional[str] = None) -> str:
        """Copy the config for a format and optional volume to ``_quarto.yml``.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume (e.g., 'vol1', 'vol2', 'vol4', 'tinytorch') for volume-specific builds

        Returns:
            Name of the config file that was copied

        Raises:
            ValueError: If format_type is not supported
            FileNotFoundError: If the config file does not exist
        """
        config_file = self.get_config_file(format_type, volume)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.activate_config_file(config_file)

        # Volume builds need a root entry point for the selected format.
        if volume:
            self._activate_index(volume, format_type)

        return config_file.name

    def activate_config_file(self, source: Path) -> None:
        """Write *source* as the active ``_quarto.yml``."""
        write_active_config(self.active_config, source, self.book_dir)

    def active_config_source(self) -> Optional[str]:
        """Return the config file the active ``_quarto.yml`` was copied from."""
        return active_config_source(self.active_config)

    def _activate_index(self, volume: str, format_type: str = "pdf") -> None:
        """Generate ``index.qmd`` from the canonical volume/format source."""
        source = write_volume_index(self.book_dir, volume, format_type)
        console.print(f"[dim]📄 Copied {source.relative_to(self.book_dir).as_posix()} → index.qmd[/dim]")

    def active_index_source(self) -> Optional[str]:
        """Return the volume index whose content ``index.qmd`` holds, or None."""
        if not self.active_index.is_file():
            return None
        content = self.active_index.read_bytes()
        for index_candidate in sorted(self.book_dir.glob("index-*.qmd")):
            if index_candidate.is_file() and index_candidate.read_bytes() == content:
                return index_candidate.relative_to(self.book_dir).as_posix()
        for vol_dir in self.book_dir.iterdir():
            if vol_dir.is_dir() and (vol_dir.name.startswith("vol") or vol_dir.name == "tinytorch"):
                for rel in ("index.qmd", "frontmatter/about.qmd"):
                    idx = vol_dir / rel
                    if idx.is_file() and idx.read_bytes() == content:
                        return idx.relative_to(self.book_dir).as_posix()
        return None

    def get_output_dir(self, format_type: str, volume: Optional[str] = None) -> Path:
        """Get the output directory from Quarto configuration.

        Args:
            format_type: Format type ('html', 'pdf', 'epub')
            volume: Optional volume (e.g., 'vol1', 'vol2', 'vol4') for volume-specific builds

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
            volume: Optional volume (e.g., 'vol1', 'vol2', 'vol4') for volume-specific builds

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

    def show_active_config(self) -> None:
        """Display which configuration and index the project is using."""
        source = self.active_config_source()
        if source:
            console.print(f"[dim]  📄 Active config: {source}[/dim]")
        elif self.active_config.exists():
            console.print("[dim]  📄 Active config: _quarto.yml (not generated by binder)[/dim]")
        else:
            console.print("[dim]  ❌ No active config found[/dim]")

        index_source = self.active_index_source()
        if index_source:
            console.print(f"[dim]  📄 Active index: {index_source}[/dim]")
        elif self.active_index.exists():
            console.print("[dim]  📄 Active index: index.qmd (matches no volume index)[/dim]")
        else:
            console.print("[dim]  ❌ No index.qmd found[/dim]")
