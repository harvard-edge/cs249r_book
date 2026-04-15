"""news list — show all drafts and published posts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from rich.table import Table

from .base import BaseCommand
from ..core.theme import Theme

try:
    import frontmatter
except ImportError:
    frontmatter = None  # Handled gracefully in run().


class ListCommand(BaseCommand):
    category = "info"

    @property
    def name(self) -> str:
        return "list"

    @property
    def description(self) -> str:
        return "Show drafts and published posts"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--drafts-only",
            action="store_true",
            help="Only show drafts",
        )
        parser.add_argument(
            "--published-only",
            action="store_true",
            help="Only show published posts",
        )

    def run(self, args: Namespace) -> int:
        if frontmatter is None:
            self.console.print(
                f"[{Theme.ERROR}]Install dependency: pip install python-frontmatter[/]"
            )
            return 1

        show_drafts = not args.published_only
        show_posts = not args.drafts_only

        if show_drafts:
            self._render_drafts()
        if show_posts:
            self._render_posts()
        return 0

    def _render_drafts(self) -> None:
        drafts = sorted(self.config.drafts_dir.glob("*.md"))
        # Exclude the template and planning docs from the drafts list.
        drafts = [
            d for d in drafts
            if d.name != "_template.md" and not d.name.startswith(("_", "ESSAY_"))
        ]

        table = Table(
            title="Drafts",
            title_style=f"{Theme.CAT_DRAFT} bold",
            show_header=True,
            header_style=Theme.SECTION,
        )
        table.add_column("File", style=Theme.EMPHASIS)
        table.add_column("Title")
        table.add_column("Category", style=Theme.DIM)

        for draft in drafts:
            try:
                post = frontmatter.load(draft)
                title = post.metadata.get("title", "—")
                cats = post.metadata.get("categories", [])
                cat = cats[0] if cats else "—"
            except Exception:
                title, cat = "(unparseable)", "—"
            table.add_row(draft.name, title, cat)

        self.console.print(table)
        self.console.print()

    def _render_posts(self) -> None:
        posts = sorted(self.config.posts_dir.rglob("*.md"), reverse=True)

        table = Table(
            title="Published",
            title_style=f"{Theme.CAT_PUBLISH} bold",
            show_header=True,
            header_style=Theme.SECTION,
        )
        table.add_column("Date", style=Theme.DIM, no_wrap=True)
        table.add_column("Title", style=Theme.EMPHASIS)
        table.add_column("Category", style=Theme.DIM)

        for post_path in posts[:15]:
            try:
                post = frontmatter.load(post_path)
                title = post.metadata.get("title", "—")
                date = post.metadata.get("date", "")
                cats = post.metadata.get("categories", [])
                cat = cats[0] if cats else "—"
            except Exception:
                title, date, cat = "(unparseable)", "", "—"
            table.add_row(str(date), title, cat)

        self.console.print(table)
