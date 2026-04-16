"""Top-level Typer entry point for the ``vault`` CLI.

Subcommands are added incrementally as phases land. At Phase 0 the app surfaces
only ``--version`` and a help panel — subcommands become real starting Phase 1.
"""

from __future__ import annotations

import typer
from rich.console import Console

from vault_cli._version import __version__
from vault_cli.exit_codes import ExitCode

console = Console()

app = typer.Typer(
    name="vault",
    help=(
        "StaffML question vault — authoring, building, and releasing the corpus.\n\n"
        "See ARCHITECTURE.md in the vault directory for full design intent."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print version and exit when ``--version`` is passed."""
    if value:
        console.print(f"vault [bold]{__version__}[/bold]")
        raise typer.Exit(code=ExitCode.SUCCESS)


@app.callback()
def main(
    _version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Entry point. Global flags live here; subcommands are attached below."""


if __name__ == "__main__":
    app()
