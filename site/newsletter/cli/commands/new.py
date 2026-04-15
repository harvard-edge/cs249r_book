"""news new — scaffold a new draft from the template."""

from __future__ import annotations

import shutil
import sys
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from ..core.console import success, info, error


class NewCommand(BaseCommand):
    category = "draft"

    @property
    def name(self) -> str:
        return "new"

    @property
    def description(self) -> str:
        return "Scaffold a new draft from the template"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "slug",
            help="URL-safe slug for the draft (e.g. 'the-builders-gap')",
        )
        parser.add_argument(
            "--category",
            choices=["essay", "community", "hands-on"],
            default="essay",
            help="Newsletter track (default: essay)",
        )

    def run(self, args: Namespace) -> int:
        template = self.config.drafts_dir / "_template.md"
        if not template.exists():
            error(f"Template not found: {template}")
            return 1

        target = self.config.drafts_dir / f"{args.slug}.md"
        if target.exists():
            error(f"Draft already exists: {target}")
            return 1

        shutil.copy(template, target)

        # Lightly seed the category in the frontmatter.
        content = target.read_text()
        content = content.replace(
            'categories: ["essay"]',
            f'categories: ["{args.category}"]',
        )
        target.write_text(content)

        success(f"Created {target.relative_to(self.config.newsletter_root.parent)}")
        info(f"Category: {args.category}")
        info("Edit the frontmatter (title, description) and start writing.")
        return 0
