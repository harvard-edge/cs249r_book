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

        if local_json:
            local_out = Path("interviews/staffml/src/data/corpus.json")
            local_result = emit_legacy_corpus(vault_dir, loaded, local_out)
            result["local_json"] = local_result
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

        # Emit vault-manifest.json on every build (was previously gated
        # behind --local-json, which caused CI publishes — which don't
        # pass that flag — to ship a stale manifest with releaseId="dev"
        # baked in from the committed file). The site reads this file
        # to label the current release, so it must always reflect the
        # release_id passed on the command line.
        manifest_out = Path("interviews/staffml/src/data/vault-manifest.json")
        # questionCount must describe what the Next.js bundle actually
        # ships — i.e., the committed corpus-summary.json — not the live
        # vault. The smoke test asserts
        # `manifest.questionCount == len(corpus-summary)`. If the vault
        # YAMLs grew since the last corpus-summary.json refresh (e.g.
        # 9525 in vault vs 9521 in bundled corpus), counting from loaded
        # would break the smoke test. Read from the bundled artifact when
        # it exists; fall back to loaded for fresh checkouts.
        #
        # chainCount stays computed from `loaded` — corpus-summary.json
        # does not carry chain memberships (those live in vault.db and
        # are served by the Worker), so reading it from corpus-summary
        # would silently drop the field to 0. The slight semantic split
        # (questionCount from bundled corpus, chainCount from live vault)
        # is acceptable because the old behavior produced both from the
        # same loaded set and chainCount has historically been "vault
        # state at build time."
        corpus_summary_path = Path(
            "interviews/staffml/src/data/corpus-summary.json"
        )
        if corpus_summary_path.exists():
            published_count = len(json.loads(corpus_summary_path.read_text()))
        else:
            published_count = sum(
                1 for lq in loaded if lq.question.status == "published"
            )
        chains_seen: set[str] = set()
        for lq in loaded:
            if lq.question.status != "published":
                continue
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
