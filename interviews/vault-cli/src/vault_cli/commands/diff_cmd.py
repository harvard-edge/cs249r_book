"""``vault diff <from> <to> [--classify]`` — compare two release artifacts (B.10).

Classifies each modification as cosmetic / semantic / structural per §4 of
ARCHITECTURE.md so maintainers can spot breaking changes before ship.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.exit_codes import ExitCode

console = Console()


@dataclass
class DiffReport:
    added:    list[str] = field(default_factory=list)
    removed:  list[str] = field(default_factory=list)
    modified: list[dict] = field(default_factory=list)


def _load_rows(db: Path) -> dict[str, dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return {
            r["id"]: {k: r[k] for k in r.keys()}
            for r in conn.execute("SELECT * FROM questions")
        }
    finally:
        conn.close()


def _classify(prev: dict, new: dict) -> str:
    """Classify a per-question change:
        - structural: topic, chain membership, status, or classification changed
        - semantic:   scenario / solution / napkin_math / deep_dive text changed
        - cosmetic:   only whitespace or inessential fields differ
    """
    structural_fields = {"topic", "track", "level", "zone", "status"}
    semantic_fields = {"scenario", "realistic_solution", "common_mistake",
                       "napkin_math", "deep_dive_url", "deep_dive_title"}

    for f in structural_fields:
        if prev.get(f) != new.get(f):
            return "structural"
    for f in semantic_fields:
        a = (prev.get(f) or "").strip() if isinstance(prev.get(f), str) else prev.get(f)
        b = (new.get(f) or "").strip() if isinstance(new.get(f), str) else new.get(f)
        # Normalize whitespace — otherwise trailing-newline diffs look like edits.
        if isinstance(a, str) and isinstance(b, str):
            a = " ".join(a.split())
            b = " ".join(b.split())
        if a != b:
            return "semantic"
    return "cosmetic"


def _diff(prev_db: Path, new_db: Path) -> DiffReport:
    prev = _load_rows(prev_db)
    new = _load_rows(new_db)
    report = DiffReport()
    report.added = sorted(set(new) - set(prev))
    report.removed = sorted(set(prev) - set(new))
    for qid in sorted(set(new) & set(prev)):
        if prev[qid] == new[qid]:
            continue
        report.modified.append({"id": qid, "classification": _classify(prev[qid], new[qid])})
    return report


def register(app: typer.Typer) -> None:
    @app.command("diff")
    def diff_cmd(
        from_version: str = typer.Argument(..., metavar="FROM"),
        to_version: str = typer.Argument(..., metavar="TO"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        classify: bool = typer.Option(True, "--classify/--no-classify",
                                      help="Label modifications as cosmetic/semantic/structural."),
        as_json: bool = typer.Option(False, "--json"),
    ) -> None:
        """Enumerate added / removed / modified questions between two releases."""
        prev = releases_dir / from_version / "vault.db"
        new = releases_dir / to_version / "vault.db"
        for p in (prev, new):
            if not p.exists():
                console.print(f"[red]error[/red]: {p} missing")
                raise typer.Exit(code=ExitCode.IO_ERROR)

        report = _diff(prev, new)

        if as_json:
            print(json.dumps({
                "ok": True,
                "exit_code": 0,
                "command": "vault diff",
                "data": {
                    "from": from_version, "to": to_version,
                    "added": [{"id": qid} for qid in report.added],
                    "removed": [{"id": qid} for qid in report.removed],
                    "modified": report.modified if classify else [
                        {"id": m["id"]} for m in report.modified
                    ],
                },
            }))
            return

        console.print(f"[cyan]{from_version}[/cyan] → [green]{to_version}[/green]")
        console.print(f"  added:    [green]+{len(report.added)}[/green]")
        console.print(f"  removed:  [red]-{len(report.removed)}[/red]")
        console.print(f"  modified: [yellow]~{len(report.modified)}[/yellow]")
        if classify and report.modified:
            buckets: dict[str, int] = {}
            for m in report.modified:
                buckets[m["classification"]] = buckets.get(m["classification"], 0) + 1
            table = Table(title="modification classification")
            table.add_column("class", style="cyan")
            table.add_column("count", justify="right")
            for cls in ("cosmetic", "semantic", "structural"):
                table.add_row(cls, str(buckets.get(cls, 0)))
            console.print(table)
            if buckets.get("structural", 0):
                console.print(
                    "[yellow]warning[/yellow]: structural changes may break "
                    "student bookmarks / chain references — review carefully."
                )
