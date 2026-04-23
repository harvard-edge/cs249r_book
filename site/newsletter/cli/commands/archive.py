"""news archive — move a sent draft from drafts/ to posts/YYYY/."""

from __future__ import annotations

import shutil
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path

from .base import BaseCommand
from ..core.console import error, info, success
from ..core.theme import Theme

try:
    import frontmatter
except ImportError:
    frontmatter = None


class ArchiveCommand(BaseCommand):
    category = "archive"

    @property
    def name(self) -> str:
        return "archive"

    @property
    def description(self) -> str:
        return "Graduate a draft to posts/YYYY/ with a date-stamped filename"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "path",
            type=Path,
            help="Path to the draft markdown file (or slug resolved against drafts/)",
        )
        parser.add_argument(
            "--date",
            default=None,
            help="Publish date in YYYY-MM-DD (default: today)",
        )
        parser.add_argument(
            "--slug",
            default=None,
            help="Filename slug (default: derived from the input filename)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview what would move, without changing anything",
        )

    def run(self, args: Namespace) -> int:
        if frontmatter is None:
            error("Install dependency: pip install python-frontmatter")
            return 1

        draft_path = self._resolve_draft_path(args.path)
        if not draft_path:
            return 1

        publish_date = args.date or date.today().isoformat()
        slug = args.slug or self._slug_from_name(draft_path.stem)
        year = publish_date.split("-")[0]
        target_dir = self.config.posts_dir / year
        target_name = f"{publish_date}_{slug}.md"
        target_path = target_dir / target_name

        self.console.print()
        info(f"Source: {draft_path.relative_to(self.config.newsletter_root.parent)}")
        info(f"Target: {target_path.relative_to(self.config.newsletter_root.parent)}")
        info(f"Publish date: {publish_date}")

        if args.dry_run:
            self.console.print()
            info("Dry run. No files moved.")
            return 0

        # Update frontmatter: remove draft flag, set date.
        post = frontmatter.load(draft_path)
        post.metadata["draft"] = False
        post.metadata["date"] = publish_date
        if "draft" in post.metadata and post.metadata["draft"] is False:
            # Keep the key False for explicitness, or remove it entirely.
            post.metadata.pop("draft", None)

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(frontmatter.dumps(post))
        draft_path.unlink()

        self.console.print()
        success(f"Published to {target_path.relative_to(self.config.newsletter_root.parent)}")
        self.console.print()
        self.console.print(
            f"[{Theme.DIM}]Suggested next step:[/] "
            f"git add {target_path.parent} && git rm {draft_path.relative_to(self.config.newsletter_root.parent)}"
        )
        return 0

    def _resolve_draft_path(self, path_arg: Path) -> Path | None:
        candidates = [path_arg]
        if not path_arg.suffix:
            candidates.append(self.config.drafts_dir / f"{path_arg.name}.md")
        if not path_arg.is_absolute() and "/" not in str(path_arg):
            candidates.append(self.config.drafts_dir / path_arg.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        error(f"Draft not found: {path_arg}")
        return None

    @staticmethod
    def _slug_from_name(stem: str) -> str:
        """Strip typical 'essay-03-the-builders-gap' → 'the-builders-gap'."""
        parts = stem.split("-", 2)
        if len(parts) == 3 and parts[0].lower() == "essay" and parts[1].isdigit():
            return parts[2]
        return stem
