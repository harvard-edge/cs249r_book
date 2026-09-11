"""Generate Quarto's root entry point from the volume's canonical source.

This standard-library-only module is also the Linux/Windows CI entry point.
"""

import argparse
from pathlib import Path
import shutil


def volume_index_source(book_dir: Path, volume: str, format_type: str) -> Path:
    """Select the homepage for HTML and the preface for PDF/EPUB."""
    if volume not in {"vol1", "vol2", "vol3", "vol4"}:
        raise ValueError(f"Unknown volume: {volume}")
    if format_type not in {"html", "pdf", "epub"}:
        raise ValueError(f"Unsupported format: {format_type}")
    if volume == "vol4":
        relative = "vol4/index.qmd" if format_type == "html" else "vol4/frontmatter/about.qmd"
    else:
        relative = f"index-{volume}.qmd"
    return Path(book_dir) / relative


def write_volume_index(book_dir: Path, volume: str, format_type: str) -> Path:
    """Refresh the ignored build copy without writing through an old symlink."""
    source = volume_index_source(book_dir, volume, format_type)
    if not source.is_file():
        raise FileNotFoundError(f"Volume entry point not found: {source}")
    active = Path(book_dir) / "index.qmd"
    if active.is_symlink():
        active.unlink()
    shutil.copyfile(source, active)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-dir", type=Path, default=Path.cwd())
    parser.add_argument("--volume", required=True, choices=("vol1", "vol2", "vol3", "vol4"))
    parser.add_argument("--format", dest="format_type", type=str.lower,
                        required=True, choices=("html", "pdf", "epub"))
    args = parser.parse_args()
    source = write_volume_index(args.book_dir, args.volume, args.format_type)
    print(f"Copied {source.relative_to(args.book_dir).as_posix()} -> index.qmd")


if __name__ == "__main__":
    main()
