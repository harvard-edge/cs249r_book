"""Shared build-artifact cleanup primitives for Binder."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console


@dataclass(frozen=True)
class ArtifactCleanupResult:
    cleaned_count: int
    restored_count: int


def clean_build_artifacts(
    book_dir: Path,
    *,
    dry_run: bool = False,
    console: Console | None = None,
) -> ArtifactCleanupResult:
    """Clean Quarto build artifacts and restore fast-build config backups."""
    out = console or Console()
    book_dir = Path(book_dir)

    out.print("[bold blue]Build Artifact Cleanup[/bold blue]")

    restored_count = 0
    for config_ext in ["_quarto-html.yml", "_quarto-pdf.yml"]:
        config_file = book_dir / "config" / config_ext
        backup_file = config_file.with_suffix(f"{config_file.suffix}.fast-build-backup")

        if backup_file.exists():
            if not dry_run:
                shutil.copy(backup_file, config_file)
                backup_file.unlink()
            restored_count += 1
            out.print(f"[green]  Restored: {config_file.name}[/green]")
        else:
            out.print(f"[dim]  Already clean: {config_file.name}[/dim]")

    artifacts_to_clean = [
        (book_dir / "_build", "Build directory (all formats)"),
        (book_dir / "index_files", "Book index files"),
        (book_dir / ".quarto", "Quarto cache (book)"),
    ]

    contents_core = book_dir / "contents" / "core"
    if contents_core.exists():
        for chapter_dir in contents_core.glob("*/"):
            if not chapter_dir.is_dir():
                continue
            for files_dir in chapter_dir.glob("*_files"):
                if not files_dir.is_dir():
                    continue
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
        if not artifact_path.exists():
            continue
        out.print(f"[yellow]  Removing: {artifact_path.name} ({description})[/yellow]")
        if not dry_run:
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()
        cleaned_count += 1

    if cleaned_count:
        out.print(f"[green]  Cleaned {cleaned_count} item(s)[/green]")
    else:
        out.print("[green]  No artifacts to clean[/green]")

    return ArtifactCleanupResult(
        cleaned_count=cleaned_count,
        restored_count=restored_count,
    )
