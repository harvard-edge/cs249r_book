"""``vault dup`` — acknowledge scenario-dedup false positives.

The LSH scenario-dedup invariant (nightly tier) flags pairs of questions
whose Jaro-Winkler similarity exceeds 0.95. Some of those are legitimate
templates (e.g., "How do you diagnose KV-cache saturation on A100/H100/H200?")
that share a common prefix. This command acknowledges such pairs so they
don't permanently red the nightly CI pipeline.

Closes Gemini R5-H-4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console

from vault_cli.exit_codes import ExitCode

console = Console()

DEFAULT_VAULT_DIR = Path("interviews/vault")


def _acks_path(vault_dir: Path) -> Path:
    # Chip R7-M-4: resolve relative to vault_dir, not CWD. Previously the
    # CWD-relative path silently missed the ack file when the CLI ran from
    # a non-default cwd — legitimate templates would red nightly CI forever.
    return vault_dir / "dedup-acks.yaml"


def _load_acks(vault_dir: Path = DEFAULT_VAULT_DIR) -> set[tuple[str, str]]:
    path = _acks_path(vault_dir)
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs = set()
    for entry in data.get("acknowledged", []) or []:
        if isinstance(entry, dict) and "a" in entry and "b" in entry:
            # Canonicalize pair order.
            a, b = sorted((entry["a"], entry["b"]))
            pairs.add((a, b))
    return pairs


def _write_acks(pairs: set[tuple[str, str]], vault_dir: Path = DEFAULT_VAULT_DIR) -> None:
    path = _acks_path(vault_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "acknowledged": [
            {"a": a, "b": b}
            for a, b in sorted(pairs)
        ],
    }
    path.write_text(
        "# Scenario near-duplicate acknowledgements.\n"
        "# Pairs listed here are excluded from the LSH dedup invariant\n"
        "# (vault check --tier slow). Add via: vault dup --ack <id-a> <id-b>\n"
        + yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def ack_pairs(vault_dir: Path | None = None) -> set[tuple[str, str]]:
    """Public helper for validator.py to read the ack list."""
    return _load_acks(vault_dir or DEFAULT_VAULT_DIR)


def register(app: typer.Typer) -> None:
    @app.command("dup")
    def dup_cmd(
        ack: tuple[str, str] | None = typer.Option(
            None, "--ack",
            help="Two question IDs (space-separated): acknowledge this pair "
                 "as intentional near-duplicates.",
        ),
        unack: tuple[str, str] | None = typer.Option(
            None, "--unack",
            help="Remove an acknowledgement so the pair is flagged again.",
        ),
        show: bool = typer.Option(False, "--show", help="List acknowledged pairs."),
        vault_dir: Path = typer.Option(
            DEFAULT_VAULT_DIR, "--vault-dir",
            help="Vault directory; determines where dedup-acks.yaml lives.",
        ),
    ) -> None:
        """Manage scenario-near-duplicate acknowledgements.

        Writes to ``<vault_dir>/dedup-acks.yaml``. The LSH dedup invariant
        in ``vault check --tier slow`` skips pairs present in this file.
        """
        pairs = _load_acks(vault_dir)
        if show:
            if not pairs:
                console.print("[dim]no acknowledged pairs[/dim]")
                return
            for a, b in sorted(pairs):
                console.print(f"  {a}  ↔  {b}")
            return

        if ack:
            a, b = sorted(ack)
            pairs.add((a, b))
            _write_acks(pairs, vault_dir)
            console.print(f"[green]acknowledged[/green]: {a} ↔ {b}")
            return

        if unack:
            a, b = sorted(unack)
            before = len(pairs)
            pairs.discard((a, b))
            if len(pairs) == before:
                console.print(f"[yellow]pair not in ack list[/yellow]: {a} ↔ {b}")
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
            _write_acks(pairs, vault_dir)
            console.print(f"[red]un-acknowledged[/red]: {a} ↔ {b}")
            return

        console.print("usage: vault dup --ack <a> <b> | --unack <a> <b> | --show", file=sys.stderr)
        raise typer.Exit(code=ExitCode.USAGE_ERROR)
