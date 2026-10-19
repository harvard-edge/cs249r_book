"""Shared build-artifact cleanup primitives for Binder."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console


@dataclass(frozen=True)
class ArtifactCleanupResult:
    """Number of artifacts removed (a dry run counts what it would remove)."""

    cleaned_count: int


def clean_build_artifacts(
    book_dir: Path,
    *,
    dry_run: bool = False,
    console: Console | None = None,
) -> ArtifactCleanupResult:
    """Remove the book's build output and Quarto caches: ``_build/``, ``index_files/``, ``.quarto/``."""
    out = console or Console()
    book_dir = Path(book_dir)

    out.print(f"[bold blue]Build Artifact Cleanup[/bold blue] [dim]({book_dir})[/dim]")

    artifacts_to_clean = [
        (book_dir / "_build", "Build directory (all formats)"),
        (book_dir / "index_files", "Book index files"),
        (book_dir / ".quarto", "Quarto cache (book)"),
    ]

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

    return ArtifactCleanupResult(cleaned_count=cleaned_count)
