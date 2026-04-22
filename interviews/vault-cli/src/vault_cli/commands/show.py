"""``vault show`` — inspect one question with its chain context.

Usage:
    vault show cloud-0185

Output:
    - full classification (track/level/zone/topic/competency_area/bloom_level/phase)
    - title + scenario preview
    - validation + human-review lineage
    - every chain the question belongs to, with prev/next walk
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vault_cli.loader import load_all


def register(app: typer.Typer) -> None:
    @app.command("show")
    def show_cmd(
        question_id: str = typer.Argument(..., help="Question ID to inspect (e.g. cloud-0185)."),
        vault_dir: Path = typer.Option(
            Path("interviews/vault"), "--vault-dir",
            help="Vault root.",
        ),
        preview_chars: int = typer.Option(400, "--preview", help="Scenario preview length (chars)."),
    ) -> None:
        """Inspect one question + walk its chains for prev/next links."""
        console = Console()
        loaded, errors = load_all(vault_dir)

        by_id = {lq.id: lq for lq in loaded}
        lq = by_id.get(question_id)
        if lq is None:
            console.print(f"[red]not found:[/red] {question_id!r}")
            raise typer.Exit(code=1)

        q = lq.question
        console.print(Panel.fit(
            f"[bold cyan]{q.id}[/bold cyan]    [dim]{q.status}[/dim]\n"
            f"[bold]{q.title}[/bold]",
            title="question",
            border_style="cyan",
        ))

        cls = Table(show_header=False, box=None, pad_edge=False)
        cls.add_column(style="dim", no_wrap=True)
        cls.add_column()
        cls.add_row("track",            q.track)
        cls.add_row("level",            q.level)
        cls.add_row("zone",             q.zone)
        cls.add_row("topic",            q.topic)
        cls.add_row("competency_area",  q.competency_area)
        cls.add_row("bloom_level",      q.bloom_level or "[dim](unset)[/dim]")
        if q.phase:
            cls.add_row("phase",        q.phase)
        cls.add_row("provenance",       q.provenance)
        console.print(cls)

        # Validation lineage.
        vs = []
        if q.validated is not None:
            mark = "✓" if q.validated else "✗"
            vs.append(f"{mark} LLM-validated ({q.validation_model or '?'}, {q.validation_date or '?'})")
        if q.math_verified is not None:
            mark = "✓" if q.math_verified else "✗"
            vs.append(f"{mark} math-verified ({q.math_model or '?'}, {q.math_date or '?'})")
        if q.human_reviewed:
            hr = q.human_reviewed
            vs.append(f"• human-reviewed: {hr.status}" + (f" by {hr.by}" if hr.by else ""))
        else:
            vs.append("[dim]• human-reviewed: not-reviewed[/dim]")
        console.print("\n[bold]Validation:[/bold]")
        for line in vs:
            console.print(f"  {line}")

        # Scenario preview.
        console.print("\n[bold]Scenario:[/bold]")
        s = q.scenario.strip()
        if len(s) > preview_chars:
            s = s[:preview_chars].rstrip() + " …"
        console.print(f"  {s}")

        # Chain memberships + prev/next walk.
        console.print("\n[bold]Chains:[/bold]")
        if not q.chains:
            console.print("  [dim](not in any chain)[/dim]")
        else:
            # Build chain → ordered member list.
            chain_members: dict[str, list] = {}
            for other in loaded:
                for c in (other.question.chains or []):
                    chain_members.setdefault(c.id, []).append((c.position, other))
            for c in sorted(q.chains, key=lambda x: x.id):
                members = sorted(chain_members.get(c.id, []), key=lambda x: x[0])
                positions = [m[0] for m in members]
                idx = next((i for i, p in enumerate(positions) if p == c.position), -1)
                console.print(f"  [cyan]{c.id}[/cyan]   position={c.position}   "
                              f"({len(members)} members, spans {sorted({m[1].question.level for m in members})})")
                prev_q = members[idx - 1][1] if idx > 0 else None
                next_q = members[idx + 1][1] if 0 <= idx < len(members) - 1 else None
                if prev_q:
                    console.print(f"    prev: [dim]{prev_q.id}[/dim]  ({prev_q.question.level}, {prev_q.question.zone})  {prev_q.question.title}")
                else:
                    console.print("    prev: [dim](start of chain)[/dim]")
                if next_q:
                    console.print(f"    next: [dim]{next_q.id}[/dim]  ({next_q.question.level}, {next_q.question.zone})  {next_q.question.title}")
                else:
                    console.print("    next: [dim](end of chain)[/dim]")

        console.print(f"\n[dim]file: {lq.path.relative_to(vault_dir)}[/dim]")


__all__ = ["register"]
