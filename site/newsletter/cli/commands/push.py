"""news push — upload a draft to Buttondown as a draft email."""

from __future__ import annotations

import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel

from .base import BaseCommand
from ..core.buttondown import ButtondownError, create_draft, upload_image
from ..core.config import load_api_key
from ..core.console import error, info, success
from ..core.theme import Theme

try:
    import frontmatter
except ImportError:
    frontmatter = None


# Match `![alt](path)` where path is NOT an absolute URL.
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(((?!https?://)[^)]+)\)")

logger = logging.getLogger(__name__)


class PushCommand(BaseCommand):
    category = "publish"

    @property
    def name(self) -> str:
        return "push"

    @property
    def description(self) -> str:
        return "Upload a draft to Buttondown (creates a draft email, does not send)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "path",
            type=Path,
            help="Path to the draft (or a slug resolved against drafts/)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would upload without pushing to Buttondown",
        )

    def run(self, args: Namespace) -> int:
        if frontmatter is None:
            error("Install dependencies: pip install python-frontmatter requests rich")
            return 1

        draft_path = self._resolve_draft_path(args.path)
        if draft_path is None:
            return 1

        post = frontmatter.load(draft_path)
        title = post.metadata.get("title")
        description = post.metadata.get("description")
        if not title:
            error(f"Draft has no `title` in frontmatter: {draft_path}")
            return 1

        self.console.print()
        self.console.print(Panel(
            f"[{Theme.EMPHASIS}]{title}[/]\n"
            f"[{Theme.DIM}]{draft_path.relative_to(self.config.newsletter_root.parent)}[/]",
            title="Pushing to Buttondown",
            border_style=Theme.CAT_PUBLISH,
            expand=False,
        ))
        self.console.print()

        if args.dry_run:
            info("Dry run: would upload the following local images")
            for match in LOCAL_IMAGE_RE.finditer(post.content):
                local_path = match.group(2).strip()
                self.console.print(f"  [{Theme.DIM}]-[/] {local_path}")
            info(f"...and create a Buttondown draft titled {title!r}")
            return 0

        api_key = load_api_key(self.config)

        # 1. Upload images.
        self.console.print(f"[{Theme.SECTION}]Images[/]")
        try:
            body = self._rewrite_images(post.content, draft_path, api_key)
        except ButtondownError as exc:
            error(str(exc))
            return 1
        if body == post.content:
            info("No local images to upload")
        self.console.print()

        # 2. Create Buttondown draft.
        self.console.print(f"[{Theme.SECTION}]Draft[/]")
        try:
            with self.console.status("Creating draft in Buttondown..."):
                email = create_draft(api_key, title, body, description)
        except ButtondownError as exc:
            error(str(exc))
            return 1

        email_id = email.get("id")
        creation_url = email.get("creation_url") or (
            f"https://buttondown.com/emails/{email_id}" if email_id else None
        )

        self.console.print()
        success("Draft created.")
        self.console.print()
        self.console.print(Panel(
            f"[{Theme.EMPHASIS}]{creation_url or 'Check buttondown.com/emails'}[/]\n\n"
            f"[{Theme.DIM}]Preview the rendering in Buttondown, then send from the UI.[/]",
            title="Preview & Send",
            border_style=Theme.BORDER_SUCCESS,
            expand=False,
        ))
        return 0

    def _resolve_draft_path(self, path_arg: Path) -> Path | None:
        """Accept a full path, a bare slug, or a draft filename."""
        candidates = [path_arg]
        if not path_arg.suffix:
            candidates.append(self.config.drafts_dir / f"{path_arg.name}.md")
        if not path_arg.is_absolute() and "/" not in str(path_arg):
            candidates.append(self.config.drafts_dir / path_arg.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        error(
            f"Draft not found. Tried:\n  "
            + "\n  ".join(str(c) for c in candidates)
        )
        return None

    def _rewrite_images(
        self, body: str, draft_path: Path, api_key: str
    ) -> str:
        """Upload every local image and replace its markdown path with the CDN URL."""
        draft_dir = draft_path.parent
        seen: dict[str, str] = {}

        def replacement(match: re.Match[str]) -> str:
            alt = match.group(1)
            local_path = match.group(2).strip()
            if local_path in seen:
                return f"![{alt}]({seen[local_path]})"

            resolved = (draft_dir / local_path).resolve()
            rel = resolved.relative_to(self.config.newsletter_root.parent)
            self.console.print(f"  [{Theme.DIM}]uploading[/] {rel}")
            cdn_url = upload_image(api_key, resolved)
            seen[local_path] = cdn_url
            self.console.print(f"  [{Theme.SUCCESS}]->[/] {cdn_url}")
            return f"![{alt}]({cdn_url})"

        return LOCAL_IMAGE_RE.sub(replacement, body)
