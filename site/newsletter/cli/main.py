"""news CLI main entry point.

Mirrors the Tito CLI pattern. argparse subcommands are registered in a
single dict (single source of truth). Each command is a class that extends
BaseCommand. All output goes through a shared Rich console with a semantic
Theme so a single file controls the color palette.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Type

from rich.panel import Panel
from rich.table import Table

from . import __version__
from .commands.archive import ArchiveCommand
from .commands.base import BaseCommand
from .commands.check import CheckCommand
from .commands.diff import DiffCommand
from .commands.list import ListCommand
from .commands.new import NewCommand
from .commands.open import OpenCommand
from .commands.pull import PullCommand
from .commands.push import PushCommand
from .commands.status import StatusCommand
from .core.config import Config
from .core.console import get_console
from .core.theme import Theme


# SINGLE SOURCE OF TRUTH: every valid subcommand lives here.
# To add a command: write a BaseCommand subclass and register it below.
COMMANDS: Dict[str, Type[BaseCommand]] = {
    "new":     NewCommand,
    "list":    ListCommand,
    "check":   CheckCommand,
    "push":    PushCommand,
    "pull":    PullCommand,
    "archive": ArchiveCommand,
    "diff":    DiffCommand,
    "open":    OpenCommand,
    "status":  StatusCommand,
}


# Category metadata drives the welcome screen grouping and coloring.
CATEGORY_META: Dict[str, tuple[str, str]] = {
    "draft":   ("Drafting",   Theme.CAT_DRAFT),
    "publish": ("Publishing", Theme.CAT_PUBLISH),
    "archive": ("Archiving",  Theme.CAT_ARCHIVE),
    "info":    ("Inspection", Theme.CAT_INFO),
}


# Exit codes. Keep narrow and meaningful.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ERROR = 1


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """Set logging level based on --verbose / --quiet flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _welcome(config: Config) -> None:
    """Render the welcome screen when `news` is run with no subcommand."""
    console = get_console()
    body = (
        f"[{Theme.EMPHASIS}]news[/] — CLI for the ML Systems newsletter.\n\n"
        f"[{Theme.DIM}]Write in the repo. Push drafts to Buttondown for preview.[/]\n"
        f"[{Theme.DIM}]Send from the Buttondown UI. Archive back to the repo.[/]"
    )
    console.print(Panel(body, border_style=Theme.BORDER_DEFAULT, expand=False))
    console.print()

    # Group registered commands by category for the help table.
    grouped: Dict[str, list[BaseCommand]] = {k: [] for k in CATEGORY_META}
    for cls in COMMANDS.values():
        cmd = cls(config)
        grouped.setdefault(cmd.category, []).append(cmd)

    for cat_key, (cat_label, cat_color) in CATEGORY_META.items():
        cmds = grouped.get(cat_key, [])
        if not cmds:
            continue
        table = Table(
            title=cat_label,
            title_style=f"{cat_color} bold",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("cmd", style=Theme.COMMAND, no_wrap=True)
        table.add_column("desc")
        for cmd in cmds:
            table.add_row(f"news {cmd.name}", cmd.description)
        console.print(table)
        console.print()

    console.print(
        f"[{Theme.DIM}]Run[/] "
        f"[{Theme.COMMAND}]news <command> --help[/] "
        f"[{Theme.DIM}]for per-command options.[/]"
    )


def _build_parser(config: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news",
        description="CLI for the ML Systems newsletter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"news {__version__}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
    )

    for name, cls in COMMANDS.items():
        cmd = cls(config)
        sub = subparsers.add_parser(
            name,
            help=cmd.description,
            description=cmd.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(_command_instance=cmd)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code."""
    # Honor NO_COLOR (https://no-color.org) — a widely adopted env var for
    # disabling ANSI color output in CLIs.
    if os.environ.get("NO_COLOR"):
        os.environ.setdefault("ANSI_COLORS_DISABLED", "1")

    argv = argv if argv is not None else sys.argv[1:]
    config = Config.discover()

    if not argv:
        _welcome(config)
        return EXIT_OK

    parser = _build_parser(config)
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    command: BaseCommand | None = getattr(args, "_command_instance", None)
    if command is None:
        _welcome(config)
        return EXIT_OK

    try:
        return command.run(args)
    except KeyboardInterrupt:
        get_console().print(f"[{Theme.WARNING}]Interrupted.[/]")
        return 130  # Conventional exit code for SIGINT
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # Unhandled exceptions should not leak a Python traceback to end
        # users; print a clean error and bail. --verbose shows the full
        # stack because logging is already configured at DEBUG.
        logging.exception("Unhandled error running `news %s`", args.command)
        get_console().print(f"[{Theme.ERROR}]error:[/] {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
