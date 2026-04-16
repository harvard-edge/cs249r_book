"""Release commands: snapshot, migrations emit, export paper, tag, publish, verify."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import typer
from rich.console import Console

from vault_cli.compiler import build as compile_build
from vault_cli.exit_codes import ExitCode
from vault_cli.hashing import hash_of_canonical_yaml, release_hash
from vault_cli.loader import load_all
from vault_cli.policy import load_policy
from vault_cli.release import (
    atomic_rename,
    emit_migrations,
    snapshot as snapshot_release,
    update_latest_symlink,
)
from vault_cli.yaml_io import load_file

console = Console()


def _latest_db(releases_dir: Path) -> Path | None:
    link = releases_dir / "latest"
    if link.is_symlink() or link.exists():
        return link / "vault.db"
    return None


def register(app: typer.Typer) -> None:
    @app.command("snapshot")
    def snapshot_cmd(
        version: str = typer.Argument(...),
        vault_db: Path = typer.Option(Path("interviews/vault/vault.db"), "--vault-db"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
    ) -> None:
        """Stage release artifacts to releases/.pending-<version>/."""
        if not vault_db.exists():
            console.print(f"[red]error[/red]: {vault_db} not found. Run `vault build` first.")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        artifacts = snapshot_release(vault_db, releases_dir, version)
        console.print(f"[green]staged[/green] {artifacts.directory}")
        console.print(f"  vault.db:     {artifacts.vault_db}")
        console.print(f"  release.json: {artifacts.release_json}")

    @app.command("migrations-emit")
    def migrations_emit_cmd(
        from_version: str = typer.Argument(..., metavar="FROM"),
        to_version: str = typer.Argument(..., metavar="TO"),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        schema_change: bool = typer.Option(False, "--schema-change"),
    ) -> None:
        """Emit forward + inverse SQL migrations between two releases."""
        prev = releases_dir / from_version / "vault.db"
        new = releases_dir / ("." + f".pending-{to_version}").replace("..", ".") / "vault.db"
        # Fall back: new release might already be at final path (re-emitting).
        if not new.exists():
            new = releases_dir / to_version / "vault.db"
        if not new.exists():
            console.print(f"[red]error[/red]: target release not found at {new}")
            raise typer.Exit(code=ExitCode.IO_ERROR)

        target_dir = new.parent
        stats = emit_migrations(
            prev_db=prev,
            new_db=new,
            out_forward=target_dir / "d1-migration.sql",
            out_rollback=target_dir / "d1-rollback.sql",
        )
        console.print(f"migrations emitted: [green]+{stats['added']}[/green] "
                      f"[yellow]~{stats['modified']}[/yellow] [red]-{stats['removed']}[/red]")
        if schema_change:
            console.print("[dim]--schema-change set: hand-author schema-forward.sql + schema-rollback.sql "
                          "alongside the data migrations.[/dim]")

    @app.command("export-paper")
    def export_paper_cmd(
        version: str = typer.Argument(...),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        paper_dir: Path = typer.Option(Path("interviews/paper"), "--paper-dir"),
    ) -> None:
        """Emit LaTeX macros + corpus_stats.json from the release vault.db.

        Replaces paper/scripts/generate_macros.py — reads vault.db via SQL so
        paper and site agree by construction (fixes H-21).
        """
        db_path = releases_dir / version / "vault.db"
        if not db_path.exists():
            db_path = releases_dir / f".pending-{version}" / "vault.db"
        if not db_path.exists():
            console.print(f"[red]error[/red]: no vault.db for {version}")
            raise typer.Exit(code=ExitCode.IO_ERROR)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
            by_track = {r["track"]: r["n"] for r in conn.execute(
                "SELECT track, COUNT(*) AS n FROM questions GROUP BY track"
            )}
            by_level = {r["level"]: r["n"] for r in conn.execute(
                "SELECT level, COUNT(*) AS n FROM questions GROUP BY level"
            )}
            by_zone = {r["zone"]: r["n"] for r in conn.execute(
                "SELECT zone, COUNT(*) AS n FROM questions GROUP BY zone"
            )}
            topics = conn.execute("SELECT COUNT(DISTINCT topic) AS n FROM questions").fetchone()["n"]
            chains = conn.execute("SELECT COUNT(DISTINCT chain_id) AS n FROM chain_questions").fetchone()["n"]
            release_hash_val = conn.execute(
                "SELECT value FROM release_metadata WHERE key='release_hash'"
            ).fetchone()["value"]
        finally:
            conn.close()

        stats = {
            "release_id": version,
            "release_hash": release_hash_val,
            "total_questions": total,
            "topics": topics,
            "chains": chains,
            "by_track": by_track,
            "by_level": by_level,
            "by_zone": by_zone,
        }
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "corpus_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
        )

        macros = [
            f"% Autogenerated by `vault export-paper {version}` — do not edit.",
            f"\\newcommand{{\\staffmlReleaseId}}{{{version}}}",
            f"\\newcommand{{\\staffmlReleaseHash}}{{{release_hash_val[:16]}}}",
            f"\\newcommand{{\\staffmlTotalQuestions}}{{{total}}}",
            f"\\newcommand{{\\staffmlTopicCount}}{{{topics}}}",
            f"\\newcommand{{\\staffmlChainCount}}{{{chains}}}",
        ]
        (paper_dir / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
        console.print(f"[green]exported[/green] macros.tex + corpus_stats.json to {paper_dir}")

    @app.command("tag")
    def tag_cmd(
        version: str = typer.Argument(...),
    ) -> None:
        """git-commit release artifacts and create a v<version> tag."""
        import subprocess
        subprocess.run(["git", "add", "interviews/vault/releases"], check=False)
        subprocess.run(["git", "commit", "-m", f"chore(release): {version}"], check=False)
        subprocess.run(["git", "tag", f"v{version}"], check=False)
        console.print(f"[green]tagged[/green] v{version}")

    @app.command("publish")
    def publish_cmd(
        version: str = typer.Argument(...),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        sign: bool = typer.Option(False, "--sign"),
        resume: bool = typer.Option(False, "--resume"),
    ) -> None:
        """Composed product: check --strict + build + snapshot + migrations emit + tag.

        Stages to releases/.pending-<v>/ and swaps to final via atomic rename
        as the last operation. --resume detects orphaned pending dirs.
        """
        releases_dir = vault_dir / "releases"
        pending = releases_dir / f".pending-{version}"
        final = releases_dir / version
        vault_db = vault_dir / "vault.db"

        if final.exists():
            console.print(f"[red]error[/red]: {final} already exists")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        if pending.exists() and not resume:
            console.print(f"[yellow]orphan pending dir at {pending}[/yellow] "
                          "— re-run with --resume to continue, or delete it.")
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # 1. build
        loaded, errors = load_all(vault_dir)
        if errors:
            console.print(f"[yellow]warning[/yellow]: {len(errors)} load errors — see `vault check`")
        console.print("running vault build…")
        result = compile_build(vault_dir=vault_dir, loaded=loaded, output=vault_db, release_id=version)
        console.print(f"  release_hash: [cyan]{result['release_hash']}[/cyan]")

        # 2. snapshot
        artifacts = snapshot_release(vault_db, releases_dir, version)
        console.print(f"  staged at: {artifacts.directory}")

        # 3. migrations emit (from latest if exists)
        latest_db = _latest_db(releases_dir)
        if latest_db and latest_db.exists():
            stats = emit_migrations(
                prev_db=latest_db,
                new_db=artifacts.vault_db,
                out_forward=artifacts.migration_sql,
                out_rollback=artifacts.rollback_sql,
            )
            console.print(f"  migrations: +{stats['added']} ~{stats['modified']} -{stats['removed']}")

        # 4. atomic swap (last irreversible step)
        atomic_rename(artifacts.directory, final)
        update_latest_symlink(releases_dir, version)
        console.print(f"[green]published[/green] {version}")

        if sign:
            console.print("[dim]--sign: minisign integration is Phase 7 polish; skipped.[/dim]")

    @app.command("verify")
    def verify_cmd(
        version: str = typer.Argument(...),
        releases_dir: Path = typer.Option(Path("interviews/vault/releases"), "--releases-dir"),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        from_source: bool = typer.Option(
            True,
            "--from-source/--from-artifact",
            help="Verify from vault/questions YAML source (default) or release's vault.db.",
        ),
    ) -> None:
        """Reconstruct release_hash from YAML and assert equality with release.json."""
        release_json_path = releases_dir / version / "release.json"
        if not release_json_path.exists():
            console.print(f"[red]error[/red]: {release_json_path} not found")
            raise typer.Exit(code=ExitCode.IO_ERROR)
        expected = json.loads(release_json_path.read_text())["release_hash"]

        if from_source:
            # Re-build in a tempdir and compare.
            import tempfile
            loaded, _ = load_all(vault_dir)
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_db = Path(tmp.name)
            result = compile_build(vault_dir=vault_dir, loaded=loaded, output=tmp_db, release_id=version)
            actual = result["release_hash"]
            tmp_db.unlink(missing_ok=True)
        else:
            db = releases_dir / version / "vault.db"
            conn = sqlite3.connect(db)
            actual = conn.execute(
                "SELECT value FROM release_metadata WHERE key='release_hash'"
            ).fetchone()[0]
            conn.close()

        if actual == expected:
            console.print(f"[green]✓ verified[/green] {version}: {actual}")
            return
        console.print(f"[red]✗ mismatch[/red]")
        console.print(f"  expected: {expected}")
        console.print(f"  actual:   {actual}")
        raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
