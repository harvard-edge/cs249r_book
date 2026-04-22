"""Top-level Typer entry point for the ``vault`` CLI.

Subcommands are added incrementally as phases land. At Phase 0 the app surfaces
only ``--version`` and a help panel — subcommands become real starting Phase 1.
"""

from __future__ import annotations

import typer
from rich.console import Console

from vault_cli._version import __version__
from vault_cli.commands import (
    authoring,
    serve_api,
)
from vault_cli.commands import (
    build as build_cmd_mod,
)
from vault_cli.commands import (
    chain as chain_cmd_mod,
)
from vault_cli.commands import (
    check as check_cmd_mod,
)
from vault_cli.commands import (
    codegen as codegen_cmd_mod,
)
from vault_cli.commands import (
    diff_cmd as diff_cmd_mod,
)
from vault_cli.commands import (
    doctor as doctor_cmd_mod,
)
from vault_cli.commands import (
    dup as dup_cmd_mod,
)
from vault_cli.commands import (
    generate as generate_cmd_mod,
)
from vault_cli.commands import (
    lint as lint_cmd_mod,
)
from vault_cli.commands import (
    ls as ls_cmd_mod,
)
from vault_cli.commands import (
    promote as promote_cmd_mod,
)
from vault_cli.commands import (
    release as release_cmd_mod,
)
from vault_cli.commands import (
    show as show_cmd_mod,
)
from vault_cli.commands import (
    stats as stats_cmd_mod,
)
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

build_cmd_mod.register(app)
check_cmd_mod.register(app)
authoring.register(app)
serve_api.register(app)
release_cmd_mod.register(app)
stats_cmd_mod.register(app)
codegen_cmd_mod.register(app)
doctor_cmd_mod.register(app)
diff_cmd_mod.register(app)
promote_cmd_mod.register(app)
dup_cmd_mod.register(app)
generate_cmd_mod.register(app)
lint_cmd_mod.register(app)
ls_cmd_mod.register(app)
show_cmd_mod.register(app)
chain_cmd_mod.register(app)


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
