"""news pull — sync sent emails from Buttondown into site/newsletter/posts/.

The inverse of `news push`. Handles three real scenarios:
    1. You edited in the Buttondown UI before sending. The repo's draft
       no longer matches what subscribers received.
    2. A collaborator sent from Buttondown without pushing via the CLI.
    3. Historical backfill of newsletters sent before this CLI existed.

Idempotent with early termination: stops after 3 consecutive unchanged
posts since Buttondown returns sent emails newest-first.

Also writes `site/newsletter/_stats.yml` (issue count, subscriber count)
so the landing page can show live counters.
"""

from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser, Namespace
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from rich.table import Table

from .base import BaseCommand
from ..core.buttondown import (
    API_BASE,
    ButtondownError,
    get_email,
    list_emails,
)
from ..core.config import load_api_key
from ..core.console import error, info, success
from ..core.theme import Theme

logger = logging.getLogger(__name__)

CONSECUTIVE_UNCHANGED_LIMIT = 3


# ---------------------------------------------------------------------------
# HTML -> plain text (for Buttondown fancy-editor bodies)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li"):
            self._pieces.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._pieces)).strip()


def _strip_html(html: str) -> str:
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _is_html(text: str) -> bool:
    return bool(re.search(r"<[a-z][^>]*>", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Email -> markdown
# ---------------------------------------------------------------------------

def _slugify(subject: str) -> str:
    slug = subject.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")


def _categorize(subject: str, body: str, default: str) -> list[str]:
    """Auto-categorize based on subject keywords, falling back to length."""
    s = subject.lower()
    b = body.lower() if body else ""

    community_kw = ("spotlight", "community update", "team newsletter",
                    "show & tell", "show and tell", "celebration",
                    "live event")
    handson_kw = ("correction", "update", "resources", "discount",
                  "applications", "kickoff", "kit", "tinytorch")

    if any(kw in s for kw in community_kw):
        return ["community"]
    if any(kw in s for kw in handson_kw):
        return ["hands-on"]

    plain = _strip_html(b) if _is_html(b) else b
    return ["essay"] if len(plain) > 1500 else [default]


def _detect_author(subject: str, body: str, fallback: str) -> str:
    """Pull a guest author out of a 'Written by X' pattern if present.

    Community/team posts default to 'MLSysBook Team' when no guest author
    is named, so a team-authored community newsletter is not mis-attributed
    to the primary author.
    """
    text = _strip_html(body) if _is_html(body) else body
    match = re.search(
        r"[Ww]ritten\s+by\s+(?:Professor\s+)?"
        r"([A-Z][a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc]+"
        r"(?:\s+[A-Z][a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc]+)+)",
        text,
    )
    if match:
        return match.group(1).strip()

    s = subject.lower()
    if any(kw in s for kw in ("team newsletter", "community update", "community spotlight")):
        return "MLSysBook Team"

    return fallback


def _parse_date(publish_date: str) -> tuple[str, str]:
    """Return (date_str, year)."""
    if publish_date:
        try:
            dt = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d"), dt.strftime("%Y")
        except ValueError:
            return publish_date[:10], publish_date[:4]
    today = datetime.utcnow()
    return today.strftime("%Y-%m-%d"), today.strftime("%Y")


def _email_to_markdown(email: dict[str, Any], default_category: str) -> tuple[str, str, str]:
    """Convert a Buttondown email to (year, filename, markdown content)."""
    subject = email.get("subject", "Untitled")
    body = email.get("body", "")
    publish_date = email.get("publish_date", "")

    categories = _categorize(subject, body, default_category)
    author = _detect_author(subject, body, fallback="Vijay Janapa Reddi")
    date_str, year = _parse_date(publish_date)
    slug = _slugify(subject)
    filename = f"{date_str}_{slug}.md"

    # Skip Buttondown's auto-generated placeholder images.
    image_url = email.get("image") or ""
    if image_url and "image-generator.buttondown.email" in image_url:
        image_url = ""

    # Strip HTML comments unconditionally before any derived field uses the
    # body. Buttondown editor metadata (the `<!-- buttondown-editor-mode:
    # fancy -->` banner, draft-status annotations, etc.) is never renderable
    # content, and a multiline comment containing `---` or Setext underlines
    # will confuse Pandoc's YAML frontmatter parser — the whole file fails to
    # render with a cryptic "multiline key may not be an implicit key" error.
    # We also want the generated `description:` field to be real prose, not
    # editor metadata leaked from the first 200 characters of the raw body.
    comment_stripped = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)

    # Clean description from comment-stripped body text.
    plain = _strip_html(comment_stripped) if _is_html(comment_stripped) else comment_stripped
    plain = plain.strip()
    if len(plain) > 200:
        description = plain[:200].rsplit(" ", 1)[0] + "..."
    else:
        description = plain
    description = description.replace('"', "'").replace("\n", " ").strip()

    # Buttondown fancy bodies are HTML; wrap for Quarto.
    if _is_html(body):
        body_block = f"\n```{{=html}}\n{comment_stripped.strip()}\n```\n"
    else:
        body_block = f"\n{comment_stripped}\n"

    image_line = f'\nimage: "{image_url}"' if image_url else ""
    content = (
        "---\n"
        f'title: "{subject.replace(chr(34), chr(39))}"\n'
        f'date: "{date_str}"\n'
        f'author: "{author}"\n'
        f'description: "{description}"\n'
        f"categories: {json.dumps(categories)}"
        f"{image_line}\n"
        "---\n"
        f"{body_block}"
    )
    return year, filename, content


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _fetch_subscriber_count(api_key: str) -> int:
    import requests
    try:
        resp = requests.get(
            f"{API_BASE}/subscribers",
            headers={"Authorization": f"Token {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("count", 0)
    except Exception:  # noqa: BLE001
        return 0


def _write_stats(stats_file: Path, email_count: int, subscriber_count: int) -> None:
    display = f"{(subscriber_count // 50) * 50}+" if subscriber_count >= 100 else str(subscriber_count)
    stats_file.write_text(
        "# Auto-generated by `news pull` — do not edit\n"
        f"issue_count: {email_count}\n"
        f"subscriber_count: {subscriber_count}\n"
        f'subscriber_display: "{display}"\n',
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

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
            help="Fallback category for short posts that do not match auto-categorization keywords",
        )
        parser.add_argument(
            "--no-stats",
            action="store_true",
            help="Skip writing site/newsletter/_stats.yml",
        )

    def run(self, args: Namespace) -> int:
        api_key = load_api_key(self.config)

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

        # Buttondown returns newest-first. Preserve that order for early termination.
        emails.sort(key=lambda e: e.get("publish_date", ""), reverse=True)

        plan = self._build_plan(emails, args)
        self._render_plan(plan, dry_run=args.dry_run)

        if args.dry_run:
            return 0

        written = updated = 0
        for action, _email, target, content in plan:
            if action in ("skip", "error"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if action == "overwrite":
                updated += 1
            else:
                written += 1

        if not args.no_stats:
            sub_count = _fetch_subscriber_count(api_key)
            stats_file = self.config.newsletter_root / "_stats.yml"
            _write_stats(stats_file, email_count=len(emails), subscriber_count=sub_count)
            info(f"Stats: {len(emails)} issues, {sub_count} subscribers -> {stats_file.name}")

        self.console.print()
        success(f"{written} new, {updated} updated")
        return 0

    # ------------------------------------------------------------------

    def _build_plan(
        self, emails: list[dict[str, Any]], args: Namespace
    ) -> list[tuple[str, dict[str, Any], Path, str]]:
        """Classify each email: new, overwrite, skip, or error."""
        plan: list[tuple[str, dict[str, Any], Path, str]] = []
        consecutive_unchanged = 0

        for email in emails:
            try:
                year, filename, content = _email_to_markdown(email, args.category)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not convert email %s: %s", email.get("id"), exc)
                plan.append(("error", email, Path(), ""))
                continue

            target = self.config.posts_dir / year / filename

            if target.exists():
                if args.force:
                    plan.append(("overwrite", email, target, content))
                    consecutive_unchanged = 0
                    continue

                # Incremental: overwrite only if remote modified after local.
                remote_mod_ts = _modification_ts(email.get("modification_date", ""))
                if remote_mod_ts and remote_mod_ts > target.stat().st_mtime:
                    plan.append(("overwrite", email, target, content))
                    consecutive_unchanged = 0
                    continue

                plan.append(("skip", email, target, ""))
                consecutive_unchanged += 1
                if consecutive_unchanged >= CONSECUTIVE_UNCHANGED_LIMIT:
                    logger.info(
                        "Stopping after %d consecutive unchanged posts",
                        CONSECUTIVE_UNCHANGED_LIMIT,
                    )
                    break
            else:
                plan.append(("new", email, target, content))
                consecutive_unchanged = 0

        return plan

    def _render_plan(
        self,
        plan: list[tuple[str, dict[str, Any], Path, str]],
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
            "overwrite": f"[{Theme.WARNING}]\u26a0 update[/]",
            "skip":      f"[{Theme.DIM}]= unchanged[/]",
            "error":     f"[{Theme.ERROR}]\u2717 error[/]",
        }
        counts = {"new": 0, "overwrite": 0, "skip": 0, "error": 0}

        for action, email, target, _content in plan:
            counts[action] += 1
            subject = email.get("subject", "(no subject)")
            date = (email.get("publish_date") or "")[:10]
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
            f"{counts['new']} new  ·  {counts['overwrite']} update  ·  "
            f"{counts['skip']} unchanged  ·  {counts['error']} error"
            f"[/]"
        )


def _modification_ts(iso: str) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
