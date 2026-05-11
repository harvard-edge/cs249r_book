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
    levels_sorted = sorted(levels, key=lambda lvl: ("L1","L2","L3","L4","L5","L6+").index(lvl))
    monotonic = levels == levels_sorted

    console.print(f"[bold cyan]{chain_id}[/bold cyan]   "
                  f"{len(members)} members   track(s)={','.join(tracks)}   topic(s)={','.join(topics)}")
    if len(topics) > 1:
        console.print("  [yellow]warning[/yellow]: chain spans multiple topics — likely mis-linked")
    if not monotonic:
        console.print("  [yellow]warning[/yellow]: levels not monotonically non-decreasing across positions")

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


@chain_app.command("audit")
def chain_audit(
    vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings",
        help="Skip embedding-based drift score (faster; structural checks only)."
    ),
    sidecar: Path = typer.Option(
        Path("interviews/vault/embeddings.npz"), "--sidecar",
        help="Path to embedding sidecar (cached per-question vectors).",
    ),
    model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", "--model",
        help="Sentence-transformer model. Calibrated default: bge-small."
    ),
) -> None:
    """Audit chain integrity. Reports orphans, position drift, stale registry, and similarity drift."""
    import json as _json

    from vault_cli.chains import audit as audit_mod
    from vault_cli.chains.embeddings import load_or_build

    corpus = audit_mod.load_corpus(vault_dir)
    store = None
    if not no_embeddings:
        store = load_or_build(list(corpus.values()), sidecar, model_name=model, progress=not json_output)

    rep = audit_mod.run_audit(vault_dir, store=store)
    if json_output:
        typer.echo(_json.dumps(audit_mod.report_to_dict(rep), indent=2, default=str))
    else:
        typer.echo(audit_mod.format_text_report(rep))


@chain_app.command("suggest")
def chain_suggest(
    vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    json_output: bool = typer.Option(False, "--json"),
    sidecar: Path = typer.Option(Path("interviews/vault/embeddings.npz"), "--sidecar"),
    model: str = typer.Option("BAAI/bge-small-en-v1.5", "--model"),
    tau_strong: float = typer.Option(0.85, "--tau-strong"),
    tau_review: float = typer.Option(0.70, "--tau-review"),
    top_k: int = typer.Option(5, "--top-k", help="Max candidates per orphan."),
) -> None:
    """Suggest rescue candidates for orphan singleton chains.

    Outputs ranked candidates within each orphan's (track, topic) bucket.
    Honors single-topic invariant + Bloom-monotonic level adjacency. Pure
    embedding-based ranking — never auto-applies.
    """
    import json as _json

    from vault_cli.chains import audit as audit_mod
    from vault_cli.chains import rescue as rescue_mod
    from vault_cli.chains.embeddings import load_or_build

    corpus = audit_mod.load_corpus(vault_dir)
    store = load_or_build(list(corpus.values()), sidecar, model_name=model, progress=not json_output)
    rescues = rescue_mod.suggest_rescues(
        vault_dir, store, tau_strong=tau_strong, tau_review=tau_review, top_k=top_k,
    )
    if json_output:
        typer.echo(_json.dumps(rescue_mod.rescues_to_dict(rescues), indent=2, default=str))
    else:
        typer.echo(rescue_mod.format_rescue_report(rescues))


def register(app: typer.Typer) -> None:
    app.add_typer(chain_app, name="chain")
    # Also expose under "chains" plural for ergonomics — both work.
    app.add_typer(chain_app, name="chains")


__all__ = ["register"]
