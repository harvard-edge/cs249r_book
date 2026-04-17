"""news diff — compare a local draft against its Buttondown version.

Useful when you've pushed a draft and then tweaked it in the Buttondown
UI. Surfaces any divergence so you can decide whether to update the repo
or re-push from the repo.
"""

from __future__ import annotations

import difflib
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

from rich.panel import Panel
from rich.syntax import Syntax

from .base import BaseCommand
from ..core.buttondown import ButtondownError, list_emails
from ..core.config import load_api_key
from ..core.console import error, info, success
from ..core.theme import Theme

try:
    import frontmatter
except ImportError:
    frontmatter = None


# Match `![alt](CDN-url)` so we can strip Buttondown CDN URLs for a fair diff.
CDN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(https?://assets\.buttondown\.email[^)]+\)")


class DiffCommand(BaseCommand):
    category = "info"

    @property
    def name(self) -> str:
        return "diff"

    @property
    def description(self) -> str:
        return "Diff a local draft against its pushed Buttondown version"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "path",
            type=Path,
            help="Path to the draft (or a slug resolved against drafts/)",
        )
        parser.add_argument(
            "--strip-images",
            action="store_true",
            help="Ignore image URL differences (local paths vs Buttondown CDN URLs)",
        )

    def run(self, args: Namespace) -> int:
        if frontmatter is None:
            error("Install dependencies: pip install -r site/newsletter/requirements.txt")
            return 1

        draft_path = self._resolve_draft_path(args.path)
        if draft_path is None:
            return 1

        post = frontmatter.load(draft_path)
        title = post.metadata.get("title")
        if not title:
            error(f"Draft has no `title`: {draft_path}")
            return 1

        api_key = load_api_key(self.config)
        try:
            drafts = list_emails(api_key, status="draft")
        except ButtondownError as exc:
            error(str(exc))
            return 1

        match = next((e for e in drafts if e.get("subject") == title), None)
        if match is None:
            error(f"No Buttondown draft matches title {title!r}. Try `news push` first.")
            return 1

        local_body = post.content
        remote_body = match.get("body", "")

        if args.strip_images:
            local_body = CDN_IMAGE_RE.sub(r"![\1](IMG)", local_body)
            remote_body = CDN_IMAGE_RE.sub(r"![\1](IMG)", remote_body)
            # Also collapse any local image references so both sides compare equal.
            local_body = re.sub(
                r"!\[([^\]]*)\]\([^)]+\)", r"![\1](IMG)", local_body
            )
            remote_body = re.sub(
                r"!\[([^\]]*)\]\([^)]+\)", r"![\1](IMG)", remote_body
            )

        diff = list(
            difflib.unified_diff(
                local_body.splitlines(keepends=True),
                remote_body.splitlines(keepends=True),
                fromfile=f"local:{draft_path.name}",
                tofile=f"buttondown:{match.get('id')}",
                lineterm="",
            )
        )

        if not diff:
            success("Local draft and Buttondown draft match.")
            info(f"Subject: {title}")
            return 0

        self.console.print()
        self.console.print(Panel(
            f"[{Theme.EMPHASIS}]{title}[/]",
            title="Diff",
            border_style=Theme.BORDER_WARNING,
            expand=False,
        ))
        diff_text = "".join(diff)
        self.console.print(Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=False))
        return 1

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
