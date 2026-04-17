"""Shared Rich console and semantic print helpers.

A single Console instance is cached at module load time so every command
prints to the same terminal with the same settings. The NO_COLOR
environment variable (https://no-color.org) is honored: setting it to any
non-empty value disables ANSI colors for the whole CLI.
"""

from __future__ import annotations

import os
from functools import lru_cache

from rich.console import Console
from rich.panel import Panel

from .theme import Theme


@lru_cache(maxsize=1)
def get_console() -> Console:
    """Return the single Rich console instance for the CLI.

    Cached with lru_cache so the same instance is returned on every call.
    This replaces the module-global pattern and is easier to test because
    we can `get_console.cache_clear()` in tests.
    """
    no_color = bool(os.environ.get("NO_COLOR"))
    return Console(
        no_color=no_color,
        highlight=False,  # Prevent Rich from auto-highlighting numbers.
        soft_wrap=False,
    )


def success(message: str) -> None:
    get_console().print(f"[{Theme.SUCCESS}]\u2713[/] {message}")


def warn(message: str) -> None:
    get_console().print(f"[{Theme.WARNING}]\u26a0[/] {message}")


def error(message: str) -> None:
    get_console().print(f"[{Theme.ERROR}]\u2717[/] {message}")


def info(message: str) -> None:
    get_console().print(f"[{Theme.INFO}]i[/] {message}")


def section(
    title: str,
    body: str,
    *,
    style: str = Theme.BORDER_DEFAULT,
) -> None:
    """Render a titled panel for grouped output."""
    get_console().print(
        Panel(body, title=title, border_style=style, expand=False)
    )
