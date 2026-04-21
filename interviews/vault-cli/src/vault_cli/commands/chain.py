"""``vault chain`` — browse and inspect question chains.

Subcommands:
    vault chain ls [--track --topic]          list chains with counts + spans
    vault chain show <chain-id>                walk a chain end-to-end

A chain links questions on a single topic into a progression, usually
across Bloom's levels. ≈32% of the corpus participates in ≥1 chain;
≈101 questions are in multiple chains.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.loader import load_all


chain_app = typer.Typer(help="Browse and inspect question chains.")


def _collect_chains(loaded) -> dict[str, list]:
    """Return {chain_id: [(position, loaded_question), ...]}."""
    out: dict[str, list] = {}
    for lq in loaded:
        for c in (lq.question.chains or []):
            out.setdefault(c.id, []).append((c.position, lq))
    for cid in out:
        out[cid].sort(key=lambda x: x[0])
    return out


@chain_app.command("ls")
def chain_ls(
    track: str | None = typer.Option(None, "--track", help="Filter by track (cloud/edge/mobile/tinyml/global)."),
    topic: str | None = typer.Option(None, "--topic", help="Filter by topic slug."),
    vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
) -> None:
    """List chains with member count + level span."""
    console = Console()
    loaded, _ = load_all(vault_dir)
    chains = _collect_chains(loaded)

    rows = []
    for cid, members in chains.items():
        topics = {m[1].question.topic for m in members}
        tracks = {m[1].question.track for m in members}
        levels = sorted({m[1].question.level for m in members})
        first_topic = next(iter(topics))
        first_track = next(iter(tracks))
        if track and first_track != track:
            continue
        if topic and topic not in topics:
            continue
        rows.append((cid, first_track, first_topic, len(members), "/".join(levels)))

    rows.sort(key=lambda r: (r[1], r[2], r[0]))

    table = Table(show_header=True, header_style="bold")
    table.add_column("chain_id", no_wrap=True, style="cyan")
    table.add_column("track", no_wrap=True)
    table.add_column("topic", no_wrap=True, style="dim")
    table.add_column("#", justify="right")
    table.add_column("level span")
    for cid, tr, tp, n, sp in rows:
        table.add_row(cid, tr, tp, str(n), sp)
    console.print(table)
    console.print(f"[dim]{len(rows)} chains[/dim]")


@chain_app.command("show")
def chain_show(
    chain_id: str = typer.Argument(..., help="Chain ID to walk (e.g. cloud-chain-432)."),
    vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
) -> None:
    """Walk a chain end-to-end: one row per member, ordered by position."""
    console = Console()
    loaded, _ = load_all(vault_dir)
    chains = _collect_chains(loaded)

    members = chains.get(chain_id)
    if members is None:
        console.print(f"[red]chain not found:[/red] {chain_id!r}")
        raise typer.Exit(code=1)

    # Detect topic/track drift within the chain.
    topics = sorted({m[1].question.topic for m in members})
    tracks = sorted({m[1].question.track for m in members})
    levels = [m[1].question.level for m in members]
    levels_sorted = sorted(levels, key=lambda L: ("L1","L2","L3","L4","L5","L6+").index(L))
    monotonic = levels == levels_sorted

    console.print(f"[bold cyan]{chain_id}[/bold cyan]   "
                  f"{len(members)} members   track(s)={','.join(tracks)}   topic(s)={','.join(topics)}")
    if len(topics) > 1:
        console.print(f"  [yellow]warning[/yellow]: chain spans multiple topics — likely mis-linked")
    if not monotonic:
        console.print(f"  [yellow]warning[/yellow]: levels not monotonically non-decreasing across positions")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("level", no_wrap=True)
    table.add_column("zone", no_wrap=True)
    table.add_column("id", no_wrap=True, style="cyan")
    table.add_column("title")
    for pos, lq in members:
        q = lq.question
        table.add_row(str(pos), q.level, q.zone, q.id, q.title)
    console.print(table)


def register(app: typer.Typer) -> None:
    app.add_typer(chain_app, name="chain")


__all__ = ["register"]
