"""Authoring primitives: new, edit, rm, restore, move.

Phase-1 minimal implementations. Each command performs the core operation with
validation and typed-confirmation safety; advanced flags (batch mode, editor
multi-file) are Phase-1.x follow-ups.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.exit_codes import ExitCode
from vault_cli.loader import load_all
from vault_cli.models import Level, Question, Track, Zone
from vault_cli.paths import classification_from_path, path_for_question, Classification
from vault_cli.yaml_io import dump_str, load_file

console = Console()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_hash(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:6]


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() or c == "-" else "-" for c in s.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-") or "untitled"


def _open_editor(path: Path) -> int:
    editor = os.environ.get("EDITOR", "vi")
    return subprocess.run([editor, str(path)], check=False).returncode


def register(app: typer.Typer) -> None:
    @app.command("new")
    def new_cmd(
        title: str = typer.Option(..., "--title", "-t", help="Question title."),
        topic: str = typer.Option(..., "--topic", help="Topic (must exist in taxonomy.yaml)."),
        track: Track = typer.Option(..., "--track"),
        level: Level = typer.Option(..., "--level"),
        zone: Zone = typer.Option(..., "--zone"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        no_edit: bool = typer.Option(False, "--no-edit", help="Skip $EDITOR."),
    ) -> None:
        """Create a new draft question with a content-addressed ID."""
        classification = Classification(track=track, level=level, zone=zone)
        topic_slug = _slug(topic)
        h = _short_hash(title)

        # Allocate seq by scanning existing files in the cell.
        cell_dir = path_for_question(vault_dir, classification, "").parent
        cell_dir.mkdir(parents=True, exist_ok=True)
        seq = 1
        while True:
            filename = f"{topic_slug}-{h}-{seq:04d}.yaml"
            candidate = cell_dir / filename
            if not candidate.exists():
                break
            seq += 1

        qid = f"{track.value}-{level.value}-{zone.value}-{topic_slug}-{h}-{seq:04d}"
        now = _now()

        payload = {
            "schema_version": 1,
            "id": qid,
            "title": title,
            "topic": topic,
            "status": "draft",
            "created_at": now,
            "last_modified": now,
            "provenance": "human",
            "scenario": "<TODO: describe the scenario in plaintext>",
            "details": {
                "realistic_solution": "<TODO: canonical answer>",
            },
        }

        candidate.write_text(dump_str(payload), encoding="utf-8")
        console.print(f"created [cyan]{candidate}[/cyan] (id={qid})")

        if not no_edit and os.environ.get("EDITOR"):
            _open_editor(candidate)

    @app.command("edit")
    def edit_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    ) -> None:
        """Open an existing question in $EDITOR. Re-validates on save."""
        loaded, errors = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
        rc = _open_editor(match.path)
        if rc != 0:
            console.print(f"[yellow]editor exited non-zero ({rc})[/yellow]")
        # Re-validate.
        try:
            Question.model_validate(load_file(match.path))
            console.print("[green]✓ validates[/green]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]validation failed[/red]: {exc}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE) from exc

    @app.command("rm")
    def rm_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        hard: bool = typer.Option(False, "--hard"),
        force: bool = typer.Option(False, "--force"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    ) -> None:
        """Soft-delete (status=deprecated) by default; --hard removes the file."""
        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        if not hard:
            data = load_file(match.path)
            data["status"] = "deprecated"
            data["last_modified"] = _now()
            match.path.write_text(dump_str(data), encoding="utf-8")
            console.print(f"[yellow]deprecated[/yellow] {match.path}")
            return

        if match.question.chain is not None and not force:
            console.print(f"[red]refusing[/red]: {question_id} is in chain {match.question.chain.id!r}; "
                          "pass --force if you really mean it")
            raise typer.Exit(code=ExitCode.USER_ABORTED)

        title = match.question.title
        confirm = typer.prompt(f"Type the full title to confirm hard delete of {question_id}")
        if confirm.strip() != title.strip():
            console.print("[yellow]aborted[/yellow]: title mismatch")
            raise typer.Exit(code=ExitCode.USER_ABORTED)
        match.path.unlink()
        console.print(f"[red]removed[/red] {match.path}")

    @app.command("restore")
    def restore_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    ) -> None:
        """Restore a deprecated question to status=published."""
        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
        data = load_file(match.path)
        data["status"] = "published"
        data["last_modified"] = _now()
        match.path.write_text(dump_str(data), encoding="utf-8")
        console.print(f"[green]restored[/green] {match.path}")

    @app.command("move")
    def move_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        to: str = typer.Option(..., "--to", help="<track>/<level>/<zone>"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        """Reclassify a question by moving its file. Uses git mv for history."""
        parts = to.split("/")
        if len(parts) != 3:
            console.print("[red]error[/red]: --to must be '<track>/<level>/<zone>'")
            raise typer.Exit(code=ExitCode.USAGE_ERROR)
        track, level, zone = parts
        classification = Classification(Track(track), Level(level), Zone(zone))

        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        target = path_for_question(vault_dir, classification, match.path.name)
        target.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            console.print(f"would git mv {match.path} {target}")
            return

        result = subprocess.run(["git", "mv", str(match.path), str(target)], check=False)
        if result.returncode != 0:
            # Fall back to plain move if not in a git repo.
            shutil.move(str(match.path), str(target))
        console.print(f"moved [cyan]{match.path}[/cyan] → [green]{target}[/green]")
