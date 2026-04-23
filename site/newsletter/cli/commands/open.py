"""news open — open a Buttondown URL in the browser.

By default, opens the Buttondown emails dashboard. Pass a slug to open
the most recent draft whose subject matches that slug (useful right
after `news push`).
"""

from __future__ import annotations

import webbrowser
from argparse import ArgumentParser, Namespace

from .base import BaseCommand
from ..core.buttondown import ButtondownError, list_emails
from ..core.config import load_api_key
from ..core.console import error, info, success

BUTTONDOWN_DASHBOARD = "https://buttondown.com/emails"


class OpenCommand(BaseCommand):
    category = "info"

    @property
    def name(self) -> str:
        return "open"

    @property
    def description(self) -> str:
        return "Open a Buttondown URL in the browser"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "target",
            nargs="?",
            default=None,
            help="Draft slug or keyword to match; omit to open the Buttondown dashboard",
        )
        parser.add_argument(
            "--archive",
            action="store_true",
            help="Open the public Buttondown archive instead of the dashboard",
        )

    def run(self, args: Namespace) -> int:
        if args.archive:
            url = "https://buttondown.com/mlsysbook/archive/"
            info(f"Opening {url}")
            webbrowser.open(url)
            return 0

        if not args.target:
            info(f"Opening {BUTTONDOWN_DASHBOARD}")
            webbrowser.open(BUTTONDOWN_DASHBOARD)
            return 0

        api_key = load_api_key(self.config)
        try:
            drafts = list_emails(api_key, status="draft")
        except ButtondownError as exc:
            error(str(exc))
            return 1

        needle = args.target.lower()
        match = next(
            (
                email
                for email in drafts
                if needle in (email.get("subject") or "").lower()
            ),
            None,
        )
        if match is None:
            error(f"No draft subject matched {args.target!r}. Try `news status`.")
            return 1

        url = match.get("creation_url") or (
            f"https://buttondown.com/emails/{match.get('id')}"
        )
        success(f"Opening {match.get('subject')!r}")
        info(url)
        webbrowser.open(url)
        return 0
