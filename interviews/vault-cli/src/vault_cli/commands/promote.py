"""``vault promote`` — move drafts to published (B.12 / §4.4).

Sets ``status: published`` and bumps ``provenance: llm-draft`` →
``llm-then-human-edited`` (since promoting implies human review). Records
the reviewer via ``--reviewed-by`` (must match the committer email per
Soumith L-NEW-1; CI enforces).
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode
from vault_cli.yaml_io import dump_str, load_file

console = Console()


def _git_user_email() -> str | None:
    try:
        res = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def _promote_one(draft_path: Path, vault_dir: Path, reviewed_by: str) -> Path:
    # Move from vault/drafts/<track>/<level>/<zone>/<file>
    # to    vault/questions/<track>/<level>/<zone>/<file>
    drafts_root = vault_dir / "drafts"
    questions_root = vault_dir / "questions"
    rel = draft_path.relative_to(drafts_root)
    target = questions_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)

    data = load_file(draft_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["status"] = "published"
    data["last_modified"] = now
    if data.get("provenance") == "llm-draft":
        data["provenance"] = "llm-then-human-edited"
        gm = data.get("generation_meta") or {}
        gm["human_reviewed_at"] = now
        data["generation_meta"] = gm
    authors = data.get("authors") or []
    if reviewed_by and reviewed_by not in authors:
        authors.append(reviewed_by)
    data["authors"] = authors

    target.write_text(dump_str(data), encoding="utf-8")

    # git mv the draft away.
    result = subprocess.run(
        ["git", "rm", "-f", str(draft_path)], capture_output=True, check=False,
    )
    if result.returncode != 0 and draft_path.exists():
        draft_path.unlink()
    return target


def register(app: typer.Typer) -> None:
    @app.command("promote")
    def promote_cmd(
        question_id: str | None = typer.Argument(None, metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        all_drafts: bool = typer.Option(False, "--all-drafts"),
        topic: str | None = typer.Option(None, "--topic", help="Filter --all-drafts by topic."),
        reviewed_by: str | None = typer.Option(
            None, "--reviewed-by",
            help="Reviewer identity (default: git config user.email). "
                 "CI rejects if this doesn't match the committer email.",
        ),
    ) -> None:
        """Promote a draft (or all drafts) to published status."""
        reviewer = reviewed_by or _git_user_email()
        if not reviewer:
            console.print(
                "[red]error[/red]: --reviewed-by not provided and git config user.email is empty"
            )
            raise typer.Exit(code=ExitCode.USAGE_ERROR)

        drafts_root = vault_dir / "drafts"
        if not drafts_root.exists():
            console.print(f"[yellow]no drafts directory[/yellow]: {drafts_root}")
            raise typer.Exit(code=ExitCode.SUCCESS)

        promoted: list[Path] = []
        if all_drafts:
            for draft in drafts_root.rglob("*.yaml"):
                if topic is not None:
                    data = load_file(draft)
                    if data.get("topic") != topic:
                        continue
                promoted.append(_promote_one(draft, vault_dir, reviewer))
        else:
            if not question_id:
                console.print("[red]usage[/red]: vault promote <id> | --all-drafts")
                raise typer.Exit(code=ExitCode.USAGE_ERROR)
            # Find the draft by id.
            for draft in drafts_root.rglob("*.yaml"):
                data = load_file(draft)
                if data.get("id") == question_id:
                    promoted.append(_promote_one(draft, vault_dir, reviewer))
                    break
            else:
                console.print(f"[red]error[/red]: draft with id {question_id!r} not found")
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        for target in promoted:
            console.print(f"[green]promoted[/green] {target}")
        if not promoted:
            console.print("[yellow]no drafts promoted[/yellow]")
