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
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from vault_cli.exit_codes import ExitCode
from vault_cli.loader import load_all
from vault_cli.models import Level, Question, Track, Zone
from vault_cli.paths import Classification, path_for_question
from vault_cli.yaml_io import dump_str, load_file

console = Console()

REGISTRY_PATH = Path("interviews/vault/id-registry.yaml")
APPLICABILITY_PATH = Path("interviews/vault/data/applicable_cells.json")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _short_hash(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:6]


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() or c == "-" else "-" for c in s.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-") or "untitled"


def _git_user_email() -> str | None:
    """Resolve the committer identity for auto-populating ``authors:`` (David H4)."""
    try:
        res = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True, check=True
        )
        email = res.stdout.strip()
        return email or None
    except subprocess.CalledProcessError:
        return None


def _working_tree_dirty() -> bool:
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(res.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def _append_registry(entry: dict) -> None:
    """Append a single ``{id, created_at, created_by}`` line to id-registry.yaml
    (append-only; fixes David H3 + C-5 enforcement)."""
    line = (
        f"  - {{id: {entry['id']}, created_at: {entry['created_at']}, "
        f"created_by: {entry['created_by']}}}\n"
    )
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(
            "# id-registry.yaml — APPEND-ONLY log. Never rewrite.\nentries:\n" + line,
            encoding="utf-8",
        )
        return
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def _inject_validation_error_comment(path: Path, error: str) -> None:
    """Prepend a YAML comment block describing the validation error so the next
    ``vault edit`` opens the file with the error visible inline (David H1)."""
    existing = path.read_text(encoding="utf-8")
    header = "# ─── VALIDATION FAILURE ──────────────────────────────────────\n"
    body_lines = [f"# {line}" for line in error.splitlines()]
    marker = "# ─────────────────────────────────────────────────────────────\n"
    # Strip any previous error block to avoid accretion.
    stripped_lines: list[str] = []
    in_block = False
    for raw in existing.splitlines(keepends=True):
        if raw.startswith("# ─── VALIDATION FAILURE"):
            in_block = True
            continue
        if in_block:
            if raw.startswith("# ────"):
                in_block = False
            continue
        stripped_lines.append(raw)
    stripped = "".join(stripped_lines)
    path.write_text(header + "\n".join(body_lines) + "\n" + marker + stripped, encoding="utf-8")


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
        skip_rebase: bool = typer.Option(
            False, "--skip-rebase",
            help="Skip the `git pull --rebase` pre-check. Use only for offline dev.",
        ),
    ) -> None:
        """Create a new draft question with a content-addressed ID.

        Appends to ``id-registry.yaml`` (append-only) and auto-populates
        ``authors`` from ``git config user.email`` (David R3-H3, R3-H4).
        Runs ``git pull --rebase`` on the registry first to reduce collision
        rate, per §3.3 concurrency contract.
        """
        classification = Classification(track=track, level=level, zone=zone)
        topic_slug = _slug(topic)
        h = _short_hash(title)

        # §3.3: git pull --rebase on the registry before allocation.
        if not skip_rebase:
            import contextlib
            with contextlib.suppress(FileNotFoundError):
                # FileNotFoundError means git isn't installed — fine for offline dev.
                subprocess.run(
                    ["git", "pull", "--rebase", "--autostash", "origin"],
                    check=False, capture_output=True,
                )

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

        author = _git_user_email()
        payload: dict = {
            "schema_version": 1,
            "id": qid,
            "title": title,
            "topic": topic,
            "status": "draft",
            "created_at": now,
            "last_modified": now,
            "provenance": "human",
        }
        if author:
            payload["authors"] = [author]
        payload.update({
            "scenario": "<TODO: describe the scenario in plaintext>",
            "details": {
                "realistic_solution": "<TODO: canonical answer>",
            },
        })

        candidate.write_text(dump_str(payload), encoding="utf-8")
        _append_registry({"id": qid, "created_at": now, "created_by": author or "unknown"})
        console.print(f"created [cyan]{candidate}[/cyan] (id={qid})")
        if author:
            console.print(f"  authors: [dim]{author}[/dim]")

        if not no_edit and os.environ.get("EDITOR"):
            _open_editor(candidate)

    @app.command("edit")
    def edit_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        retries: int = typer.Option(
            3, "--retries",
            help="Max re-open attempts on validation failure (David R3-H1).",
        ),
    ) -> None:
        """Open an existing question in $EDITOR. On validation failure, inject
        an error-comment block at the top of the file and re-open for iteration
        — preserving the authoring flow instead of raising a terminal error
        (David R3-H1 authoring-UX fix).
        """
        # Path may not exist if we're editing a file that failed to load (e.g.,
        # because it has an injected error block). Look by both loaded-match and
        # by parsing each candidate's ``id:`` field directly.
        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if match:
            target_path = match.path
        else:
            # Chip R4-M-1 fix: parse each YAML's `id:` field for an EXACT match
            # rather than substring-matching the file body (which could open the
            # wrong file if the id appears in a chain reference elsewhere).
            # Strip any prior validation-error header comments so bypassed
            # files still parse.
            target_path = None
            import re as _re
            id_re = _re.compile(r"^\s*id:\s*['\"]?([A-Za-z0-9._-]+)['\"]?\s*$", _re.MULTILINE)
            for p in (vault_dir / "questions").rglob("*.yaml"):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                m = id_re.search(text)
                if m and m.group(1) == question_id:
                    target_path = p
                    break
            if target_path is None:
                console.print(f"[red]error[/red]: id not found: {question_id}")
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        attempts = 0
        while attempts < retries:
            rc = _open_editor(target_path)
            if rc != 0:
                console.print(f"[yellow]editor exited non-zero ({rc}) — aborting[/yellow]")
                raise typer.Exit(code=ExitCode.USER_ABORTED)
            try:
                Question.model_validate(load_file(target_path))
                console.print("[green]✓ validates[/green]")
                return
            except (ValidationError, Exception) as exc:  # noqa: BLE001
                attempts += 1
                msg = str(exc)
                console.print(f"[red]validation failed[/red] (attempt {attempts}/{retries}): "
                              f"injecting error block at top of file")
                _inject_validation_error_comment(target_path, msg)
                if attempts >= retries:
                    console.print(f"[red]giving up after {retries} attempts[/red]; "
                                  f"fix the YAML and re-run `vault edit {question_id}`")
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
        allow_dirty: bool = typer.Option(False, "--allow-dirty", help="Skip dirty-tree refusal."),
        force_chain: bool = typer.Option(
            False, "--i-understand-chain-breakage",
            help="Permit moving a chained question out of its chain.",
        ),
    ) -> None:
        """Reclassify a question by moving its file. Uses git mv for history.

        Refuses on dirty tree (David R3-H2), on chain-breakage, and on target
        cells excluded by the applicability matrix — matching the §4.1 contract.
        """
        parts = to.split("/")
        if len(parts) != 3:
            console.print("[red]error[/red]: --to must be '<track>/<level>/<zone>'")
            raise typer.Exit(code=ExitCode.USAGE_ERROR)
        track, level, zone = parts
        classification = Classification(Track(track), Level(level), Zone(zone))

        if not allow_dirty and _working_tree_dirty():
            console.print(
                "[red]refusing[/red]: working tree is dirty. "
                "Commit or stash first, or pass --allow-dirty."
            )
            raise typer.Exit(code=ExitCode.USER_ABORTED)

        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # Chain-breakage check: refuse if question is in a chain unless user opts in.
        if match.question.chain is not None and not force_chain:
            console.print(
                f"[red]refusing[/red]: {question_id} is in chain {match.question.chain.id!r}. "
                "Pass --i-understand-chain-breakage to proceed (breaks chain integrity)."
            )
            raise typer.Exit(code=ExitCode.USER_ABORTED)

        # Applicability-matrix check: refuse if (track, topic) is excluded.
        if APPLICABILITY_PATH.exists():
            import json
            matrix = json.loads(APPLICABILITY_PATH.read_text())
            excluded = {
                (cell["track"], cell["topic"])
                for cell in matrix.get("excluded_cells", []) or []
            }
            if (track, match.question.topic) in excluded:
                console.print(
                    f"[red]refusing[/red]: ({track!r}, {match.question.topic!r}) "
                    "is in the excluded-cells set per applicable_cells.json."
                )
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

    @app.command("renumber")
    def renumber_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    ) -> None:
        """Recover from a dedup-seq collision by bumping the seq suffix and
        renaming + updating the id field (David R3-N-2 resolution in code).

        Post-rebase workflow: if a rebase/merge lands another PR's file at
        your seq slot, run ``vault renumber <old-id>`` to allocate the next
        free seq. Appends a new registry entry; the old ID is still reserved
        (never reused).
        """
        if _working_tree_dirty():
            console.print("[red]refusing[/red]: dirty working tree. Commit or stash first.")
            raise typer.Exit(code=ExitCode.USER_ABORTED)

        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # Parse old filename: <topic>-<hash>-<seq>.yaml
        import re
        m = re.match(r"(?P<tk>[a-z0-9-]+?)-(?P<h>[a-f0-9]{6})-(?P<seq>\d{4})\.yaml$", match.path.name)
        if not m:
            console.print(f"[red]error[/red]: filename doesn't match expected content-addressed pattern: {match.path.name}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
        topic_kebab = m.group("tk")
        h = m.group("h")
        cell_dir = match.path.parent

        # Allocate next free seq.
        seq = int(m.group("seq")) + 1
        while True:
            new_name = f"{topic_kebab}-{h}-{seq:04d}.yaml"
            if not (cell_dir / new_name).exists():
                break
            seq += 1

        # Build new id — same prefix, new seq.
        parts = question_id.rsplit("-", 1)
        new_id = f"{parts[0]}-{seq:04d}"
        new_path = cell_dir / new_name

        data = load_file(match.path)
        data["id"] = new_id
        data["last_modified"] = _now()
        new_path.write_text(dump_str(data), encoding="utf-8")
        subprocess.run(["git", "rm", "-f", str(match.path)], check=False, capture_output=True)
        subprocess.run(["git", "add", str(new_path)], check=False, capture_output=True)

        _append_registry({
            "id": new_id, "created_at": _now(),
            "created_by": _git_user_email() or "unknown",
        })
        console.print(f"[green]renumbered[/green] {question_id} → {new_id}")
        console.print(f"  old: [dim]{match.path}[/dim]")
        console.print(f"  new: [cyan]{new_path}[/cyan]")

    @app.command("mark-exemplar")
    def mark_exemplar_cmd(
        question_id: str = typer.Argument(..., metavar="ID"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
    ) -> None:
        """Promote a question to the curated exemplar pool used by `vault generate`.

        Refuses unless provenance is ``human`` or
        ``llm-then-human-edited`` with ``human_reviewed_at`` set — per §12.2
        (David R3-N-9 resolution in code). For external PRs, CI gates on a
        maintainer-approval label ``exemplar-approved``.
        """
        loaded, _ = load_all(vault_dir)
        match = next((lq for lq in loaded if lq.id == question_id), None)
        if not match:
            console.print(f"[red]error[/red]: id not found: {question_id}")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        prov = match.question.provenance.value
        if prov == "human":
            pass  # eligible
        elif prov == "llm-then-human-edited":
            gm = match.question.generation_meta
            if gm is None or gm.human_reviewed_at is None:
                console.print(
                    "[red]refusing[/red]: llm-then-human-edited requires "
                    "generation_meta.human_reviewed_at to be set."
                )
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
        else:
            console.print(
                f"[red]refusing[/red]: provenance={prov!r} is not exemplar-eligible "
                "(must be 'human' or 'llm-then-human-edited' with human_reviewed_at)."
            )
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        exemplars_root = vault_dir / "exemplars"
        rel = match.path.relative_to(vault_dir / "questions")
        target = exemplars_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "mv", str(match.path), str(target)], check=False, capture_output=True,
        )
        if result.returncode != 0:
            shutil.move(str(match.path), str(target))
        console.print(f"[green]marked as exemplar[/green]: {target}")
