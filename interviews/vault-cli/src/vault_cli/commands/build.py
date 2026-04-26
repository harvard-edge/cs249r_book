"""``vault build`` — compile vault/questions/ → vault.db."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.compiler import build as compile_build
from vault_cli.exit_codes import ExitCode
from vault_cli.legacy_export import copy_visual_assets, emit_legacy_corpus, emit_manifest
from vault_cli.loader import load_all

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("build")
    def build_cmd(
        vault_dir: Path = typer.Option(
            Path("interviews/vault"),
            "--vault-dir",
            help="Path to vault/ directory (default: interviews/vault).",
        ),
        output: Path = typer.Option(
            Path("interviews/vault/vault.db"),
            "--output",
            "-o",
            help="Output SQLite path.",
        ),
        release_id: str = typer.Option("dev", "--release-id", help="Release identifier to stamp."),
        as_json: bool = typer.Option(False, "--json", help="Emit machine-readable summary."),
        legacy_json: bool = typer.Option(
            False,
            "--legacy-json",
            help="Also regenerate the site-compatible corpus.json at "
                 "interviews/staffml/src/data/corpus.json from YAML. "
                 "Required until Phase-4 cutover; closes §11.1 'corpus.json "
                 "is generated, not authored'.",
        ),
    ) -> None:
        """Compile all YAML questions under vault/questions/ to a SQLite file.

        Reports validation failures but continues; the resulting vault.db
        contains all questions that passed Pydantic validation and policy.
        """
        loaded, errors = load_all(vault_dir)
        if errors:
            console.print(f"[yellow]warning[/yellow]: {len(errors)} load/validation errors skipped")
            for err in errors[:10]:
                console.print(f"  [dim]{err.path}[/dim]: {err.message}")

        if not loaded:
            console.print("[red]error[/red]: no questions loaded")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        result = compile_build(
            vault_dir=vault_dir,
            loaded=loaded,
            output=output,
            release_id=release_id,
        )

        if legacy_json:
            legacy_out = Path("interviews/staffml/src/data/corpus.json")
            legacy_result = emit_legacy_corpus(vault_dir, loaded, legacy_out)
            result["legacy_json"] = legacy_result
            console.print(
                f"[dim]legacy corpus.json: {legacy_result['count']} questions → "
                f"{legacy_result['output']}[/dim]"
            )
            # Mirror visual assets alongside the JSON. The frontend
            # references /question-visuals/<track>/<file>.svg directly
            # from Next.js's public/ tree — no hydration or worker
            # round-trip, just static assets cached at the edge.
            visuals_out = Path("interviews/staffml/public")
            visuals_result = copy_visual_assets(vault_dir, visuals_out)
            result["visual_assets"] = visuals_result
            console.print(
                f"[dim]visual assets: {visuals_result.get('total_assets', 0)} total, "
                f"{visuals_result.get('copied', 0)} copied, "
                f"{visuals_result.get('deleted', 0)} pruned → "
                f"{visuals_out}/question-visuals/[/dim]"
            )
            # E.2: emit vault-manifest.json deterministically from the
            # same loaded set that produced corpus.json. Eliminates the
            # recurring stale-manifest pre-commit failure (the manifest
            # was previously hand-maintained).
            manifest_out = Path("interviews/staffml/src/data/vault-manifest.json")
            # Count chains from the corpus.json the legacy emitter just
            # wrote — chain shape there is the canonical published view.
            chains_seen: set[str] = set()
            for lq in loaded:
                if lq.question.status != "published":
                    continue
                for ch in (lq.question.chains or []):
                    chains_seen.add(ch.id)
            manifest_result = emit_manifest(
                loaded=loaded,
                output=manifest_out,
                release_id=str(result.get("release_id", "dev")),
                release_hash=str(result.get("release_hash", "")),
                published_count=int(legacy_result["count"]),
                chain_count=len(chains_seen),
            )
            result["manifest"] = manifest_result
            console.print(
                f"[dim]vault-manifest.json: {manifest_result['questionCount']} "
                f"questions / {manifest_result['chainCount']} chains → "
                f"{manifest_result['output']}[/dim]"
            )

        if as_json:
            print(json.dumps({
                "ok": True,
                "exit_code": 0,
                "exit_symbol": "SUCCESS",
                "command": "vault build",
                "data": result,
            }))
            return

        table = Table(title="vault build")
        table.add_column("key", style="cyan")
        table.add_column("value")
        for k, v in result.items():
            table.add_row(k, str(v))
        console.print(table)
        if errors:
            console.print(f"[dim]({len(errors)} records skipped — see above)[/dim]")
