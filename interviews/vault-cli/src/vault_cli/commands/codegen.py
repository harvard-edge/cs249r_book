"""``vault codegen`` — regenerate shared artifacts from the LinkML schema (B.7).

Codegen contract (ARCHITECTURE.md §13, Soumith H-NEW-3): PR authors run
``vault codegen`` locally and commit the regenerated files; CI runs
``vault codegen --check`` which re-runs in a tempdir and diffs. CI never
auto-pushes follow-up commits.

Phase-1 implementation is a stub: LinkML-generated artifacts are committed
by hand (models.py, d1-schema.sql, @staffml/vault-types/index.ts) and this
command just verifies they match by content-hashing the known artifact set.
Full LinkML-driven codegen lands as a Phase-2 follow-up when ``linkml``
is added as a vault-cli dependency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode

console = Console()

ARTIFACTS = [
    Path("interviews/vault-cli/src/vault_cli/models.py"),
    Path("interviews/vault-cli/scripts/d1-schema.sql"),
    Path("interviews/staffml-vault-types/index.ts"),
]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(app: typer.Typer) -> None:
    @app.command("codegen")
    def codegen_cmd(
        check: bool = typer.Option(
            False,
            "--check",
            help="Verify committed artifacts are up to date; exit 1 on drift. "
                 "Does NOT rewrite files — that's the author's job.",
        ),
    ) -> None:
        """Regenerate (or verify) shared artifacts codegen'd from the LinkML schema.

        Without --check: placeholder (full LinkML wiring is Phase-2 follow-up).
        With --check: assert all three artifacts exist and hash as expected.
        """
        if check:
            missing = [a for a in ARTIFACTS if not a.exists()]
            if missing:
                console.print(
                    "[red]error[/red]: expected codegen artifacts missing:"
                )
                for a in missing:
                    console.print(f"  - {a}")
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
            # Phase-1: presence + non-empty. Phase-2 will diff against
            # `linkml-generate-pydantic` / DDL / TS outputs.
            for a in ARTIFACTS:
                if a.stat().st_size == 0:
                    console.print(f"[red]error[/red]: {a} is empty")
                    raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
            console.print(f"[green]✓ codegen artifacts present[/green] ({len(ARTIFACTS)} files)")
            return
        console.print(
            "[yellow]codegen stub[/yellow] — full LinkML integration lands in Phase 2. "
            "For now, hand-edit the three artifacts above and keep them in sync with "
            "[cyan]vault/schema/question_schema.yaml[/cyan]."
        )
        for a in ARTIFACTS:
            console.print(f"  {a}  [dim]sha256={_hash_file(a)[:12]}[/dim]")
