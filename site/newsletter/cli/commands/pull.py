"""news pull — sync sent emails from Buttondown into site/newsletter/posts/.

The inverse of `news push`. Reconciles three real scenarios:
    1. You edited in the Buttondown UI before sending. The repo's draft
       no longer matches what subscribers received.
    2. A collaborator sent from Buttondown without pushing via the CLI.
    3. Historical backfill of newsletters sent before this CLI existed.

Idempotent: a post that already exists in posts/YYYY/ is left alone
unless --force is passed.
"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from .base import BaseCommand
from ..core.buttondown import ButtondownError, get_email, list_emails
from ..core.config import load_api_key
from ..core.console import error, info, success, warn
from ..core.theme import Theme

try:
    import frontmatter
except ImportError:
    frontmatter = None

logger = logging.getLogger(__name__)


class PullCommand(BaseCommand):
    category = "publish"

    @property
    def name(self) -> str:
        return "pull"

    @property
    def description(self) -> str:
        return "Sync sent emails from Buttondown into posts/YYYY/"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "email_id",
            nargs="?",
            default=None,
            help="Optional Buttondown email id; if omitted, sync all sent emails",
        )
        parser.add_argument(
            "--since",
            default=None,
            help="Only pull emails sent on or after this date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be pulled without writing any files",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing posts (default: skip)",
        )
        parser.add_argument(
            "--category",
            default="update",
            choices=["essay", "community", "hands-on", "update"],
            help="Default category to assign pulled posts (default: update)",
        )

    def run(self, args: Namespace) -> int:
        if frontmatter is None:
            error("Install dependencies: pip install -r site/newsletter/requirements.txt")
            return 1

        api_key = load_api_key(self.config)

        # 1. Fetch the emails to consider.
        try:
            if args.email_id:
                emails = [get_email(api_key, args.email_id)]
            else:
                with self.console.status("Fetching sent emails from Buttondown..."):
                    emails = list_emails(api_key, status="sent")
        except ButtondownError as exc:
            error(str(exc))
            return 1

        if args.since:
            emails = [e for e in emails if (e.get("publish_date") or "")[:10] >= args.since]

        if not emails:
            info("No sent emails found.")
            return 0

        # 2. Classify each one: new, skipped (exists), or conflicted.
        results: list[tuple[str, dict[str, Any], Path]] = []
        for email in emails:
            target = self._target_path(email, args.category)
            if not target:
                results.append(("error", email, Path()))
                continue
            if target.exists() and not args.force:
                results.append(("skip", email, target))
            else:
                results.append(("new" if not target.exists() else "overwrite", email, target))

        # 3. Render the plan.
        self._render_plan(results, dry_run=args.dry_run)

        if args.dry_run:
            return 0

        # 4. Write files.
        written = 0
        for action, email, target in results:
            if action in ("skip", "error"):
                continue
            self._write_post(email, target, args.category)
            written += 1

        self.console.print()
        success(f"{written} post(s) written to posts/")
        return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_path(self, email: dict[str, Any], default_category: str) -> Path | None:
        """Compute the canonical posts/YYYY/YYYY-MM-DD_slug.md path."""
        date = (email.get("publish_date") or email.get("creation_date") or "")[:10]
        subject = email.get("subject") or ""
        if not date or not subject:
            return None
        year = date.split("-")[0]
        slug = _slugify(subject)
        return self.config.posts_dir / year / f"{date}_{slug}.md"

    def _write_post(self, email: dict[str, Any], target: Path, category: str) -> None:
        """Serialize a Buttondown email as a markdown post with frontmatter."""
        target.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(
            email.get("body", ""),
            title=email.get("subject", ""),
            date=(email.get("publish_date") or email.get("creation_date") or "")[:10],
            author=email.get("author_name") or "Vijay Janapa Reddi",
            categories=[category],
            description=(email.get("description") or "").strip(),
        )
        image = _first_image_url(email.get("body", ""))
        if image:
            post.metadata["image"] = image
        target.write_text(frontmatter.dumps(post))

    def _render_plan(
        self,
        results: list[tuple[str, dict[str, Any], Path]],
        *,
        dry_run: bool,
    ) -> None:
        table = Table(
            title="Pull plan" + (" (dry run)" if dry_run else ""),
            title_style=f"{Theme.CAT_PUBLISH} bold",
            show_header=True,
            header_style=Theme.SECTION,
        )
        table.add_column("Action", no_wrap=True)
        table.add_column("Date", style=Theme.DIM, no_wrap=True)
        table.add_column("Subject", style=Theme.EMPHASIS)
        table.add_column("Target", style=Theme.DIM, overflow="fold")

        style = {
            "new":       f"[{Theme.SUCCESS}]\u2713 new[/]",
            "overwrite": f"[{Theme.WARNING}]\u26a0 overwrite[/]",
            "skip":      f"[{Theme.DIM}]= exists[/]",
            "error":     f"[{Theme.ERROR}]\u2717 missing date/subject[/]",
        }

        counts = {"new": 0, "overwrite": 0, "skip": 0, "error": 0}
        for action, email, target in results:
            counts[action] += 1
            subject = email.get("subject", "(no subject)")
            date = (email.get("publish_date") or email.get("creation_date") or "")[:10]
            target_str = (
                str(target.relative_to(self.config.newsletter_root.parent))
                if target and target != Path()
                else ""
            )
            table.add_row(style[action], date, subject, target_str)
        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print(
            f"[{Theme.DIM}]"
            f"{counts['new']} new  ·  {counts['overwrite']} overwrite  ·  "
            f"{counts['skip']} skip  ·  {counts['error']} error"
            f"[/]"
        )


def _slugify(subject: str) -> str:
    """Convert a subject line into a kebab-case slug (max 80 chars)."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", subject).strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned[:80].strip("-")


def _first_image_url(body: str) -> str | None:
    """Find the first image URL in a markdown body, if any."""
    match = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", body)
    return match.group(1) if match else None
