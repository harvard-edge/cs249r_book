"""``vault ls`` — browse questions with axis filters.

One-line-per-question output, aligned columns, filterable on every
first-class classification axis. Powers the "I want to see level at a
glance" workflow without opening individual YAMLs.

Usage:
    vault ls                             # every question in the vault
    vault ls --track cloud               # cloud-only
    vault ls --level L6+                 # only L6+ questions
    vault ls --zone mastery              # only mastery-zone
    vault ls --topic kv-cache-management
    vault ls --status published
    vault ls --in-chains                 # only questions that are in >=1 chain
    vault ls --track cloud --level L4 --zone diagnosis   # combinable

Output columns: id | track | level | zone | topic | #chains | title
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.loader import load_all


def register(app: typer.Typer) -> None:
    @app.command("ls")
    def ls_cmd(
        track: str | None = typer.Option(None, "--track", help="Filter by track."),
        level: str | None = typer.Option(None, "--level", help="Filter by level (L1 … L6+)."),
        zone: str | None = typer.Option(None, "--zone", help="Filter by zone (one of the 11 ikigai zones)."),
        topic: str | None = typer.Option(None, "--topic", help="Filter by topic slug."),
        status: str | None = typer.Option(None, "--status", help="Filter by status (published, draft, flagged, archived, deleted)."),
        in_chains: bool = typer.Option(False, "--in-chains", help="Only questions that belong to ≥1 chain."),
        vault_dir: Path = typer.Option(
            Path("interviews/vault"), "--vault-dir",
            help="Vault root.",
        ),
        limit: int = typer.Option(0, "--limit", "-n", help="Truncate output after N rows (0 = all)."),
        plain: bool = typer.Option(False, "--plain", help="Plain tab-separated output (for piping to awk/grep)."),
    ) -> None:
        """List questions with axis filters. Aligned output or TSV."""
        console = Console()
        loaded, errors = load_all(vault_dir)
        if errors:
            console.print(f"[yellow]warning[/yellow]: {len(errors)} load errors (skipped)")

        rows = []
        for lq in loaded:
            q = lq.question
            if track and q.track != track: continue
            if level and q.level != level: continue
            if zone and q.zone != zone: continue
            if topic and q.topic != topic: continue
            if status and q.status != status: continue
            n_chains = len(q.chains or [])
            if in_chains and n_chains == 0: continue
            rows.append((q.id, q.track, q.level, q.zone, q.topic, n_chains, q.title))

        rows.sort(key=lambda r: (r[1], r[2], r[0]))   # track, level, id
        if limit:
            rows = rows[:limit]

        if plain:
            for r in rows:
                print("\t".join(str(x) for x in r))
            console.print(f"[dim]({len(rows)} rows)[/dim]", style="dim")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("id", no_wrap=True, style="cyan")
        table.add_column("track", no_wrap=True)
        table.add_column("level", no_wrap=True)
        table.add_column("zone", no_wrap=True)
        table.add_column("topic", no_wrap=True, style="dim")
        table.add_column("ch", justify="right", style="dim")
        table.add_column("title", no_wrap=False)
        for qid, tr, lv, zn, tp, nc, ti in rows:
            table.add_row(qid, tr, lv, zn, tp, str(nc), ti)
        console.print(table)
        console.print(f"[dim]{len(rows)} rows[/dim]")


__all__ = ["register"]
