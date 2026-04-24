"""news set-author — attach a byline to a Buttondown email's metadata.

The byline lives in ``email.metadata.author`` on Buttondown. ``news pull``
reads this field first (before any body-text heuristics), so once set,
the byline is authoritative and survives body edits and re-pulls.

Matches a local .md post against a Buttondown email by (a) exact email
id, (b) Buttondown slug, or (c) substring of the subject. Prints a
dry-run plan by default; pass ``--apply`` to actually PATCH.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any

from rich.table import Table

from .base import BaseCommand
from ..core.buttondown import (
    ButtondownError,
    get_email,
    list_emails,
    update_email,
)
from ..core.config import load_api_key
from ..core.console import error, info, success
from ..core.theme import Theme

logger = logging.getLogger(__name__)


def _resolve_target(api_key: str, ref: str) -> dict[str, Any]:
    """Resolve a user-provided reference to a single Buttondown email.

    `ref` may be:
      - a full Buttondown email id (starts with 'em_')
      - a slug (matched exactly against email.slug)
      - a substring of the subject (case-insensitive, must be unique)
    """
    if ref.startswith("em_"):
        return get_email(api_key, ref)

    # Otherwise fetch the catalog and match locally.
    emails = list_emails(api_key, status="sent")

    exact_slug = [e for e in emails if e.get("slug") == ref]
    if len(exact_slug) == 1:
        return exact_slug[0]
    if len(exact_slug) > 1:
        raise ButtondownError(f"Slug '{ref}' matched {len(exact_slug)} emails; use an id")

    needle = ref.lower()
    subject_matches = [e for e in emails if needle in (e.get("subject") or "").lower()]
    if len(subject_matches) == 1:
        return subject_matches[0]
    if len(subject_matches) > 1:
        raise ButtondownError(
            f"Subject substring '{ref}' matched {len(subject_matches)} emails; "
            "be more specific or use an id"
        )
    raise ButtondownError(f"No email matched '{ref}'")


class SetAuthorCommand(BaseCommand):
    category = "publish"

    @property
    def name(self) -> str:
        return "set-author"

    @property
    def description(self) -> str:
        return "Set metadata.author on a Buttondown email (the byline shown on the website)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "ref",
            help="Email id (em_...), slug, or unique substring of the subject",
        )
        parser.add_argument(
            "author",
            help='Byline to store, e.g. "Marco Zennaro" or "MLSysBook Team"',
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Actually PATCH the email (default: dry run — show the plan only)",
        )

    def run(self, args: Namespace) -> int:
        api_key = load_api_key(self.config)

        try:
            email = _resolve_target(api_key, args.ref)
        except ButtondownError as exc:
            error(str(exc))
            return 1

        existing_meta = email.get("metadata") or {}
        existing_author = existing_meta.get("author", "")
        new_author = args.author.strip()

        table = Table(
            title="Set author" + ("" if args.apply else " (dry run)"),
            title_style=f"{Theme.CAT_PUBLISH} bold",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Email id", email.get("id", ""))
        table.add_row("Subject", email.get("subject", ""))
        table.add_row("Slug", email.get("slug", ""))
        table.add_row("Current metadata.author", existing_author or "(none)")
        table.add_row("New metadata.author", new_author)
        self.console.print(table)

        if existing_author == new_author:
            info("No change needed — already set.")
            return 0

        if not args.apply:
            info("Dry run. Pass --apply to PATCH the email.")
            return 0

        # Merge: preserve any other metadata keys, overwrite just 'author'.
        merged = {**existing_meta, "author": new_author}
        try:
            update_email(api_key, email["id"], {"metadata": merged})
        except ButtondownError as exc:
            error(f"PATCH failed: {exc}")
            return 1

        success(f"Set metadata.author = {new_author!r} on {email['id']}")
        return 0
