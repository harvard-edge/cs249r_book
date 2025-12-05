import argparse
import shutil
from pathlib import Path

from rich.console import Console


def clean_artifacts(book_dir_str: str, dry_run: bool = False):
    """
    Clean build artifacts and restore configs.
    """
    console = Console()
    book_dir = Path(book_dir_str)

    console.print("[bold blue]🧹 Build Artifact Cleanup[/bold blue]")

    # Restore configs
    for config_ext in ["_quarto-html.yml", "_quarto-pdf.yml"]:
        config_file = book_dir / "config" / config_ext
        backup_file = config_file.with_suffix(f"{config_file.suffix}.fast-build-backup")

        if backup_file.exists():
            if not dry_run:
                shutil.copy(backup_file, config_file)
                backup_file.unlink()
            console.print(f"[green]  ✅ Restored: {config_file.name}[/green]")
        else:
            console.print(f"[dim]  - Already clean: {config_file.name}[/dim]")

    # Define artifacts to clean
    artifacts_to_clean = [
        (book_dir / "_build", "Build directory (all formats)"),
        (book_dir / "index_files", "Book index files"),
        (book_dir / ".quarto", "Quarto cache (book)"),
    ]

    # Clean Quarto-generated figure directories
    contents_core = book_dir / "contents" / "core"
    if contents_core.exists():
        for chapter_dir in contents_core.glob("*/"):
            if chapter_dir.is_dir():
                for files_dir in chapter_dir.glob("*_files"):
                    if files_dir.is_dir():
                        figure_html_dir = files_dir / "figure-html"
                        if figure_html_dir.exists():
                            artifacts_to_clean.append(
                                (
                                    figure_html_dir,
                                    f"Quarto figure artifacts ({chapter_dir.name})",
                                )
                            )

                figure_html_direct = chapter_dir / "figure-html"
                if figure_html_direct.exists():
                    artifacts_to_clean.append(
                        (
                            figure_html_direct,
                            f"Quarto figure artifacts ({chapter_dir.name})",
                        )
                    )

    cleaned_count = 0
    for artifact_path, description in artifacts_to_clean:
        if artifact_path.exists():
            console.print(
                f"[yellow]  🗑️  Removing: {artifact_path.name} ({description})[/yellow]"
            )
            if not dry_run:
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                else:
                    artifact_path.unlink()
            cleaned_count += 1

    if cleaned_count > 0:
        console.print(f"[green]  ✅ Cleaned {cleaned_count} items successfully[/green]")
    else:
        console.print("[green]  ✅ No artifacts to clean[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean build artifacts from the Quarto project."
    )
    parser.add_argument(
        "--book-dir",
        type=str,
        default="quarto",
        help="Path to the book directory (default: 'quarto').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually deleting anything.",
    )
    args = parser.parse_args()

    clean_artifacts(book_dir_str=args.book_dir, dry_run=args.dry_run)
