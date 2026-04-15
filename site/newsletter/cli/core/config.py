"""Filesystem layout and credentials for the news CLI.

Resolved lazily via `Config.discover()`: the CLI's own location on disk
tells us where the newsletter root is. There is no global config state;
callers pass a Config instance down explicitly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved filesystem layout for the newsletter CLI.

    Attributes:
        newsletter_root: site/newsletter/
        drafts_dir:      site/newsletter/drafts/
        posts_dir:       site/newsletter/posts/
        cli_dir:         site/newsletter/cli/
        env_file:        site/newsletter/cli/.env
    """

    newsletter_root: Path
    drafts_dir: Path
    posts_dir: Path
    cli_dir: Path
    env_file: Path

    @classmethod
    def discover(cls) -> "Config":
        """Find the newsletter root by walking up from this file.

        Relies on the CLI living at `site/newsletter/cli/core/config.py`.
        If that layout ever changes, update `_walk_up` below.
        """
        cli_dir = Path(__file__).resolve().parent.parent
        newsletter_root = cli_dir.parent
        return cls(
            newsletter_root=newsletter_root,
            drafts_dir=newsletter_root / "drafts",
            posts_dir=newsletter_root / "posts",
            cli_dir=cli_dir,
            env_file=cli_dir / ".env",
        )


def load_api_key(config: Config) -> str:
    """Load BUTTONDOWN_API_KEY from the shell environment or .env file.

    Precedence:
        1. BUTTONDOWN_API_KEY environment variable
        2. BUTTONDOWN_API_KEY= line in site/newsletter/cli/.env

    Exits with a helpful message if neither is set. The .env file is
    gitignored; the .env.example template is what we commit.
    """
    env_key = os.environ.get("BUTTONDOWN_API_KEY")
    if env_key:
        return env_key

    if config.env_file.exists():
        for line in config.env_file.read_text().splitlines():
            if line.startswith("BUTTONDOWN_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value

    sys.exit(
        "BUTTONDOWN_API_KEY is not set.\n"
        "Either export it in your shell:\n"
        "    export BUTTONDOWN_API_KEY=your-key-here\n"
        "Or create a .env file:\n"
        f"    cp {config.env_file.with_suffix('.example')} {config.env_file}\n"
        "    # then edit .env to add your key\n"
        "Get your key from https://buttondown.com/settings/programming"
    )
