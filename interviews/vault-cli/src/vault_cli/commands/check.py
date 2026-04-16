"""``vault check`` — run invariant checks against vault/questions/."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode
from vault_cli.loader import load_all
from vault_cli.validator import fast_tier, run_all, structural_tier

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("check")
    def check_cmd(
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        strict: bool = typer.Option(False, "--strict", help="Run fast + structural tiers (CI default)."),
        tier: str = typer.Option("all", "--tier", help="fast | structural | all"),
        as_json: bool = typer.Option(False, "--json"),
    ) -> None:
        """Run vault invariants. Exit 0 on pass, 1 on any failure."""
        loaded, load_errors = load_all(vault_dir)

        failures = []
        if tier in {"fast", "all"} or strict:
            failures.extend(fast_tier(loaded, vault_dir))
        if tier in {"structural", "all"} or strict:
            failures.extend(structural_tier(loaded, vault_dir))
        if tier == "all" and not strict:
            failures = run_all(loaded, vault_dir)

        # Load errors are always reported as failures.
        total_fail = len(failures) + len(load_errors)

        if as_json:
            errors = [
                {
                    "uri": str(e.path),
                    "severity": 1,
                    "code": "yaml-load-or-schema",
                    "source": "vault-check",
                    "message": e.message,
                }
                for e in load_errors
            ] + [
                {
                    "uri": str(f.path) if f.path else "",
                    "severity": 1,
                    "code": f.check,
                    "source": f"vault-check-{f.tier}",
                    "message": f.message,
                }
                for f in failures
            ]
            print(json.dumps({
                "ok": total_fail == 0,
                "exit_code": 0 if total_fail == 0 else 1,
                "exit_symbol": "SUCCESS" if total_fail == 0 else "VALIDATION_FAILURE",
                "command": "vault check",
                "data": {
                    "loaded": len(loaded),
                    "load_errors": len(load_errors),
                    "invariant_failures": len(failures),
                },
                "errors": errors,
            }))
            raise typer.Exit(code=ExitCode.SUCCESS if total_fail == 0 else ExitCode.VALIDATION_FAILURE)

        console.print(f"loaded [cyan]{len(loaded)}[/cyan] questions; "
                      f"[red]{len(load_errors)}[/red] load errors; "
                      f"[red]{len(failures)}[/red] invariant failures")
        for e in load_errors[:20]:
            console.print(f"  [dim]{e.path}[/dim]: {e.message}")
        for f in failures[:20]:
            tag = f"[{f.tier}/{f.check}]"
            where = f.question_id or (str(f.path) if f.path else "")
            console.print(f"  [red]{tag}[/red] {where}: {f.message}")
        if total_fail == 0:
            console.print("[green]✓ all invariants passed[/green]")
        raise typer.Exit(code=ExitCode.SUCCESS if total_fail == 0 else ExitCode.VALIDATION_FAILURE)
