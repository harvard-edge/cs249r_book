"""Generate Quarto's root entry point from the volume's canonical source.

This standard-library-only module is also the Linux/Windows CI entry point.
"""

import argparse
import os
import re
from pathlib import Path


def volume_index_source(book_dir: Path, volume: str, format_type: str) -> Path:
    """Resolve the canonical entry point file for a volume and build format.

    Selects the homepage for HTML and the preface or landing file for PDF/EPUB.
    Checks for legacy root-level ``index-{volume}.qmd`` first; if absent,
    resolves dynamically from within the volume directory:
    - HTML: ``{volume}/index.qmd`` (falling back to ``{volume}/frontmatter/about.qmd``).
    - PDF/EPUB: ``{volume}/frontmatter/about.qmd`` (falling back to ``{volume}/index.qmd``).

    Args:
        book_dir: Path to the books root directory.
        volume: Volume identifier (e.g., "vol1", "vol4", "tinytorch").
        format_type: Build format ("html", "pdf", or "epub").

    Returns:
        Path to the resolved canonical source file.

    Raises:
        ValueError: If format_type is not one of {"html", "pdf", "epub"}.
    """
    if format_type not in {"html", "pdf", "epub"}:
        raise ValueError(f"Unsupported format: {format_type}")
    book_dir = Path(book_dir)

    # If root-level index-{volume}.qmd exists (canonical for legacy vol1-vol3)
    root_index = book_dir / f"index-{volume}.qmd"
    if root_index.is_file():
        return root_index

    # Otherwise resolve from volume directory (e.g. vol4, volN, tinytorch)
    if format_type == "html":
        html_idx = book_dir / volume / "index.qmd"
        if html_idx.is_file():
            return html_idx
        about_idx = book_dir / volume / "frontmatter" / "about.qmd"
        if about_idx.is_file():
            return about_idx
        return html_idx
    else:
        about_idx = book_dir / volume / "frontmatter" / "about.qmd"
        if about_idx.is_file():
            return about_idx
        idx = book_dir / volume / "index.qmd"
        if idx.is_file():
            return idx
        return about_idx


def volume_index_bytes(book_dir: Path, source: Path) -> bytes:
    """Return canonical entry-point bytes with includes relative to the generated root.

    This pure preparation step is shared with active-source detection so relocation
    does not make Binder lose the generated document's provenance.
    """
    content = source.read_bytes()
    if source.parent != Path(book_dir):
        # Quarto resolves include paths against the generated root document.
        # Keep volume-owned includes pointing at their original files after copying.
        def relocate_include(match: re.Match[str]) -> str:
            argument = match.group("path")
            quote = argument[0] if argument.startswith(("\"", "'")) else ""
            target = argument[1:-1] if quote else argument
            if target.startswith("/") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                return match.group(0)
            relative = os.path.relpath(source.parent / target, Path(book_dir)).replace(os.sep, "/")
            return match.group("prefix") + quote + relative + quote + match.group("suffix")

        include = re.compile(
            r"(?P<prefix>\{\{<\s*include\s+)(?P<path>\"[^\"]+\"|'[^']+'|[^\s>]+)(?P<suffix>\s*>\}\})"
        )
        content = include.sub(relocate_include, content.decode("utf-8")).encode("utf-8")
    return content


def write_volume_index(book_dir: Path, volume: str, format_type: str) -> Path:
    """Refresh the root ``index.qmd`` copy without writing through an existing symlink.

    Finds the canonical index source via ``volume_index_source``, ensures any
    stale symlink at ``book_dir / "index.qmd"`` is removed, and copies the source
    file to ``book_dir / "index.qmd"``. Relative include paths are relocated when
    the canonical source lives in a subdirectory; its original bytes stay untouched.

    Args:
        book_dir: Path to the books root directory.
        volume: Volume identifier (e.g., "vol1", "vol4", "tinytorch").
        format_type: Build format ("html", "pdf", or "epub").

    Returns:
        Path to the source file that was copied into ``index.qmd``.

    Raises:
        FileNotFoundError: If the resolved source file does not exist on disk.
    """
    source = volume_index_source(book_dir, volume, format_type)
    if not source.is_file():
        raise FileNotFoundError(f"Volume entry point not found: {source}")
    active = Path(book_dir) / "index.qmd"
    content = volume_index_bytes(Path(book_dir), source)
    if active.exists() or active.is_symlink():
        active.unlink()
    active.write_bytes(content)
    return source


def main() -> None:
    """CLI entry point for standalone index synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-dir", type=Path, default=Path.cwd())
    parser.add_argument("--volume", required=True, help="Volume name (e.g. vol1, vol2, vol4, etc.)")
    parser.add_argument("--format", dest="format_type", type=str.lower,
                        required=True, choices=("html", "pdf", "epub"))
    args = parser.parse_args()
    source = write_volume_index(args.book_dir, args.volume, args.format_type)
    print(f"Copied {source.relative_to(args.book_dir).as_posix()} -> index.qmd")


if __name__ == "__main__":
    main()
