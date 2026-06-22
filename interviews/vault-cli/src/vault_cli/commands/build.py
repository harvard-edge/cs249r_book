"""``vault build`` — compile vault/questions/ → vault.db."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.compiler import build as compile_build
from vault_cli.exit_codes import ExitCode
from vault_cli.legacy_export import (
    copy_visual_assets,
    emit_corpus_summary,
    emit_legacy_corpus,
    emit_manifest,
    select_release_items,
)
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
        local_json: bool = typer.Option(
            False,
            "--local-json",
            "--local",
            help="Materialize the local-dev artifacts so the StaffML frontend "
                 "can serve full question content from disk: writes "
                 "interviews/staffml/src/data/corpus.json AND mirrors it to "
                 "interviews/staffml/public/data/corpus.json (the path the "
                 "Next.js loader actually fetches with "
                 "NEXT_PUBLIC_VAULT_FALLBACK=static). Production never reads "
                 "either file; this is dev-only. The shorter --local alias "
                 "is preferred.",
        ),
        site_bundle: bool = typer.Option(
            False,
            "--site-bundle",
            help="Materialize only the production StaffML bundle artifacts: "
                 "src/data/corpus-summary.json and public/question-visuals/. "
                 "Does not emit the heavy local-dev corpus.json.",
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
        release_items = select_release_items(vault_dir, loaded)

        site_bundle_count: int | None = None
        if local_json:
            local_out = Path("interviews/staffml/src/data/corpus.json")
            local_result = emit_legacy_corpus(vault_dir, loaded, local_out)
            result["local_json"] = local_result
            site_bundle_count = int(local_result["count"])
            console.print(
                f"[dim]local corpus.json: {local_result['count']} questions → "
                f"{local_result['output']}[/dim]"
            )
            # Mirror corpus.json into public/data/ so Next can serve it as a
            # static asset. The frontend's getStaticFullDetail() fetches
            # /data/corpus.json (set NEXT_PUBLIC_VAULT_FALLBACK=static to
            # opt in) — Turbopack does not bundle the src/data/ copy because
            # it would balloon the prod bundle, so the public mirror is the
            # only reliable runtime path in local dev.
            public_out = Path("interviews/staffml/public/data/corpus.json")
            public_out.parent.mkdir(parents=True, exist_ok=True)
            public_out.write_bytes(local_out.read_bytes())
            console.print(
                f"[dim]public mirror:    {local_result['count']} questions → "
                f"{public_out}[/dim]"
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
        elif site_bundle:
            summary_out = Path("interviews/staffml/src/data/corpus-summary.json")
            summary_result = emit_corpus_summary(vault_dir, loaded, summary_out)
            result["site_bundle"] = summary_result
            site_bundle_count = int(summary_result["count"])
            console.print(
                f"[dim]corpus-summary.json: {summary_result['count']} questions → "
                f"{summary_result['output']}[/dim]"
            )
            visuals_out = Path("interviews/staffml/public")
            visuals_result = copy_visual_assets(vault_dir, visuals_out)
            result["visual_assets"] = visuals_result
            console.print(
                f"[dim]visual assets: {visuals_result.get('total_assets', 0)} total, "
                f"{visuals_result.get('copied', 0)} copied, "
                f"{visuals_result.get('deleted', 0)} pruned → "
                f"{visuals_out}/question-visuals/[/dim]"
            )

        # The site imports this manifest at build time for release labels and
        # corpus counts, so every build stamps it from the current release id.
        manifest_out = Path("interviews/staffml/src/data/vault-manifest.json")
        # questionCount must match the policy-filtered release set. When this
        # command writes frontend artifacts, verify their count before stamping
        # the manifest so bundle drift cannot be hidden by a later smoke test.
        published_count = (
            site_bundle_count
            if site_bundle_count is not None
            else int(result["published_count"])
        )
        if published_count != len(release_items):
            console.print(
                "[red]error[/red]: generated site bundle count does not match "
                f"release policy ({published_count} != {len(release_items)})"
            )
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
        chains_seen: set[str] = set()
        for lq in release_items:
            for ch in (lq.question.chains or []):
                chains_seen.add(ch.id)
        manifest_result = emit_manifest(
            loaded=loaded,
            output=manifest_out,
            release_id=str(result["release_id"]),
            release_hash=str(result["release_hash"]),
            schema_version="1",
            policy_version=str(result["policy_version"]),
            published_count=published_count,
            chain_count=len(chains_seen),
            release_items=release_items,
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
