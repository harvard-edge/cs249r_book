"""news status — show Buttondown drafts and recently sent emails."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace

from rich.table import Table

from .base import BaseCommand
from ..core.buttondown import ButtondownError, list_emails
from ..core.config import load_api_key
from ..core.console import error, info
from ..core.theme import Theme

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10


class StatusCommand(BaseCommand):
    category = "info"

    @property
    def name(self) -> str:
        return "status"

    @property
    def description(self) -> str:
        return "Show Buttondown drafts and recently sent emails"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--limit",
            type=int,
            default=DEFAULT_LIMIT,
            help=f"Max rows per section (default: {DEFAULT_LIMIT})",
        )
        parser.add_argument(
            "--drafts-only",
            action="store_true",
            help="Only show drafts",
        )
        parser.add_argument(
            "--sent-only",
            action="store_true",
            help="Only show recently sent",
        )

    def run(self, args: Namespace) -> int:
        api_key = load_api_key(self.config)

        show_drafts = not args.sent_only
        show_sent = not args.drafts_only

        if show_drafts:
            try:
                drafts = list_emails(api_key, status="draft")
            except ButtondownError as exc:
                error(str(exc))
                return 1
            self._render("Buttondown drafts", drafts[: args.limit], Theme.CAT_DRAFT)

        if show_sent:
            try:
                sent = list_emails(api_key, status="sent")
            except ButtondownError as exc:
                error(str(exc))
                return 1
            self._render("Recently sent", sent[: args.limit], Theme.CAT_PUBLISH)

        return 0

    def _render(self, title: str, rows: list[dict], color: str) -> None:
        if not rows:
            info(f"{title}: none")
            return

        table = Table(
            title=title,
            title_style=f"{color} bold",
            show_header=True,
            header_style=Theme.SECTION,
        )
        table.add_column("Subject", style=Theme.EMPHASIS)
        table.add_column("Created", style=Theme.DIM)
        table.add_column("URL", style=Theme.DIM, overflow="fold")

        for row in rows:
            subject = row.get("subject", "—")
            created = (row.get("creation_date") or row.get("created") or "")[:10]
            url = row.get("absolute_url") or row.get("url") or ""
            table.add_row(subject, str(created), url)

        self.console.print(table)
        self.console.print()
