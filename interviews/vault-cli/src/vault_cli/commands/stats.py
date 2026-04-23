"""``vault stats`` — scorecard over vault.db (B.8).

Also wires the ``--exemplar-coverage`` audit from scripts/exemplar_coverage_audit.py
into the CLI surface (ARCHITECTURE.md §14 Phase 0 milestone; Chip R3-H3).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.exit_codes import ExitCode

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("stats")
    def stats_cmd(
        vault_db: Path = typer.Option(Path("interviews/vault/vault.db"), "--vault-db"),
        as_json: bool = typer.Option(False, "--json"),
        prometheus: bool = typer.Option(False, "--format-prometheus", help="Emit Prometheus scrape-ready metrics."),
        exemplar_coverage: bool = typer.Option(
            False, "--exemplar-coverage",
            help="Run the exemplar-coverage audit over corpus.json (Phase 0 artifact).",
        ),
    ) -> None:
        """Scorecard over the release. Fast path for dashboards + paper stats."""
        if exemplar_coverage:
            # Delegate to the scripts/ one-shot.
            script = Path(__file__).resolve().parents[3] / "scripts" / "exemplar_coverage_audit.py"
            if not script.exists():
                console.print(f"[red]error[/red]: {script} missing")
                raise typer.Exit(code=ExitCode.IO_ERROR)
            result = subprocess.run([sys.executable, str(script)], check=False)
            raise typer.Exit(code=result.returncode)

        if not vault_db.exists():
            console.print(f"[red]error[/red]: {vault_db} not found — run `vault build` first")
            raise typer.Exit(code=ExitCode.IO_ERROR)

        conn = sqlite3.connect(vault_db)
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
            by_status = {r["status"]: r["n"] for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM questions GROUP BY status"
            )}
            by_track = {r["track"]: r["n"] for r in conn.execute(
                "SELECT track, COUNT(*) AS n FROM questions GROUP BY track"
            )}
            by_provenance = {r["provenance"]: r["n"] for r in conn.execute(
                "SELECT provenance, COUNT(*) AS n FROM questions GROUP BY provenance"
            )}
            topics = conn.execute("SELECT COUNT(DISTINCT topic) AS n FROM questions").fetchone()["n"]
            chains = conn.execute("SELECT COUNT(DISTINCT chain_id) AS n FROM chain_questions").fetchone()["n"]
            meta = {r["key"]: r["value"] for r in conn.execute(
                "SELECT key, value FROM release_metadata"
            )}
        finally:
            conn.close()

        data = {
            "release_id": meta.get("release_id"),
            "release_hash": meta.get("release_hash"),
            "total": total,
            "topics": topics,
            "chains": chains,
            "by_status": by_status,
            "by_track": by_track,
            "by_provenance": by_provenance,
        }

        if as_json:
            print(json.dumps({"ok": True, "data": data}, sort_keys=True))
            return

        if prometheus:
            lines = [
                f'vault_questions_total {total}',
                f'vault_topics_total {topics}',
                f'vault_chains_total {chains}',
            ]
            for track, n in by_track.items():
                lines.append(f'vault_questions_by_track{{track="{track}"}} {n}')
            for prov, n in by_provenance.items():
                lines.append(f'vault_questions_by_provenance{{provenance="{prov}"}} {n}')
            print("\n".join(lines))
            return

        table = Table(title=f"vault stats — release {data['release_id']}")
        table.add_column("metric", style="cyan")
        table.add_column("value")
        table.add_row("total questions", str(total))
        table.add_row("topics", str(topics))
        table.add_row("chains", str(chains))
        for status, n in sorted(by_status.items()):
            table.add_row(f"status:{status}", str(n))
        for track, n in sorted(by_track.items()):
            table.add_row(f"track:{track}", str(n))
        for prov, n in sorted(by_provenance.items()):
            table.add_row(f"provenance:{prov}", str(n))
        console.print(table)
