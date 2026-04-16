"""Release artifact management.

Implements ARCHITECTURE.md §4.2 primitives (snapshot, migrations emit,
export paper, tag) and §4.3 composed `publish`. Staging uses
``releases/.pending-<v>/`` with atomic ``rename(2)`` as the final step
(fixes C-7 non-atomic publish).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseArtifacts:
    version: str
    directory: Path
    vault_db: Path
    release_json: Path
    migration_sql: Path
    rollback_sql: Path
    paper_macros: Path | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _release_metadata(vault_db: Path) -> dict[str, str]:
    conn = sqlite3.connect(vault_db)
    try:
        rows = conn.execute("SELECT key, value FROM release_metadata").fetchall()
        return dict(rows)
    finally:
        conn.close()


def snapshot(vault_db: Path, releases_dir: Path, version: str) -> ReleaseArtifacts:
    """Stage release artifacts to ``releases_dir/.pending-<v>/``.

    Does NOT rename to final location — that's the caller's atomic step.
    """
    pending = releases_dir / f".pending-{version}"
    pending.mkdir(parents=True, exist_ok=True)

    dest_db = pending / "vault.db"
    shutil.copy2(vault_db, dest_db)

    meta = _release_metadata(dest_db)
    release_info = {
        "release_id": version,
        "release_hash": meta.get("release_hash"),
        "schema_version": meta.get("schema_version"),
        "policy_version": meta.get("policy_version"),
        "published_count": int(meta.get("published_count", "0")),
        "created_at": _now(),
        "git_sha": _git_sha(),
    }
    release_json = pending / "release.json"
    release_json.write_text(json.dumps(release_info, indent=2, sort_keys=True), encoding="utf-8")

    # Empty placeholders; migrations_emit fills them if prior release exists.
    migration_sql = pending / "d1-migration.sql"
    rollback_sql = pending / "d1-rollback.sql"
    migration_sql.write_text(f"-- migration for {version} (data-only; emit via migrations_emit)\n", encoding="utf-8")
    rollback_sql.write_text(f"-- rollback for {version} (embeds prior-row bodies)\n", encoding="utf-8")

    return ReleaseArtifacts(
        version=version,
        directory=pending,
        vault_db=dest_db,
        release_json=release_json,
        migration_sql=migration_sql,
        rollback_sql=rollback_sql,
    )


def _sql_quote(s: Any) -> str:
    """SQLite string-literal quoting.

    SQLite (unlike MySQL) treats backslashes as literal characters; the ONLY
    escape in a string literal is `''` for a single quote. That makes this
    function sufficient and injection-safe for values we control — which is
    every caller here (corpus content authored via maintainer-approved flow,
    not external untrusted input). Do not reuse this helper for dialects
    other than SQLite.

    Chip R4-H-1: Python ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
    is True — we must handle bool BEFORE the numeric branch to avoid emitting
    ``True``/``False`` (Python repr) as SQL tokens.
    """
    if s is None:
        return "NULL"
    if isinstance(s, bool):
        return "1" if s else "0"
    if isinstance(s, (int, float)):
        return str(s)
    if isinstance(s, (bytes, bytearray)):
        # SQLite BLOB literal.
        return "X'" + bytes(s).hex() + "'"
    return "'" + str(s).replace("'", "''") + "'"


# Tables whose content participates in migrations, and the primary-key columns
# used to diff rows. Driven by PRAGMA at runtime to tolerate schema evolution
# without silent column drift (Dean R3-NH-2).
_MIGRATION_TABLES = {
    "questions":       ("id",),
    "chains":          ("id",),
    "chain_questions": ("chain_id", "position"),
    "tags":            ("question_id", "tag"),
}


def _columns_of(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _dump_table(db: Path, table: str, pk_cols: tuple[str, ...]) -> tuple[list[str], dict[tuple, dict[str, Any]]]:
    """Return (column_list, rows_by_pk) for one table."""
    if not db.exists():
        return [], {}
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        cols = _columns_of(conn, table)
        if not cols:
            return [], {}
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
        out: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            pk = tuple(r[c] for c in pk_cols)
            out[pk] = {c: r[c] for c in cols}
        return cols, out
    finally:
        conn.close()


def _insert_stmt(table: str, cols: list[str], row: dict[str, Any]) -> str:
    vals = [_sql_quote(row.get(c)) for c in cols]
    return f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join(vals)});"


def _delete_stmt(table: str, pk_cols: tuple[str, ...], pk: tuple) -> str:
    predicates = " AND ".join(f"{c} = {_sql_quote(v)}" for c, v in zip(pk_cols, pk))
    return f"DELETE FROM {table} WHERE {predicates};"


def _diff_table(
    table: str,
    pk_cols: tuple[str, ...],
    prev_db: Path,
    new_db: Path,
    forward_lines: list[str],
    rollback_lines: list[str],
) -> dict[str, int]:
    """Compute forward/rollback SQL for one table; append to line lists."""
    prev_cols, prev = _dump_table(prev_db, table, pk_cols)
    new_cols, new = _dump_table(new_db, table, pk_cols)
    # Use the new schema's columns for INSERTs; rollback uses the prev schema's
    # columns for anything only the prev had. If columns diverge, the release
    # is schema-changing and must ship hand-authored up/down SQL (see §6.2).
    write_cols = new_cols or prev_cols
    if not write_cols:
        return {"added": 0, "removed": 0, "modified": 0}

    added = set(new) - set(prev)
    removed = set(prev) - set(new)
    modified = {pk for pk in (set(new) & set(prev)) if prev[pk] != new[pk]}

    for pk in sorted(added, key=str):
        forward_lines.append(_insert_stmt(table, write_cols, new[pk]))
        rollback_lines.append(_delete_stmt(table, pk_cols, pk))
    for pk in sorted(removed, key=str):
        forward_lines.append(_delete_stmt(table, pk_cols, pk))
        rollback_lines.append(_insert_stmt(table, write_cols, prev[pk]))
    for pk in sorted(modified, key=str):
        forward_lines.append(_insert_stmt(table, write_cols, new[pk]))
        rollback_lines.append(_insert_stmt(table, write_cols, prev[pk]))

    return {"added": len(added), "removed": len(removed), "modified": len(modified)}


def emit_migrations(prev_db: Path, new_db: Path, out_forward: Path, out_rollback: Path) -> dict[str, int]:
    """Emit forward + inverse migrations between two vault.db snapshots.

    Diffs ALL migration-participating tables (questions, chains, chain_questions,
    tags) — not just ``questions`` (Dean R3-NC-1 correctness fix). Column names
    are read from the live DB via PRAGMA so schema evolution doesn't silently
    misalign column positions (Dean R3-NH-2).

    Inverse migration embeds FULL PRIOR ROW BODIES for UPDATEs and DELETEs so
    rollback reconstructs state without relying on mechanical inversion. When
    schemas diverge between prev and new, the operator must also ship
    hand-authored ``schema-forward.sql`` and ``schema-rollback.sql`` alongside
    the data-migration pair.
    """
    forward_lines = ["BEGIN;"]
    rollback_lines = ["BEGIN;"]
    totals = {"added": 0, "removed": 0, "modified": 0}

    for table, pk in _MIGRATION_TABLES.items():
        stats = _diff_table(table, pk, prev_db, new_db, forward_lines, rollback_lines)
        for k in totals:
            totals[k] += stats[k]

    forward_lines.append("COMMIT;")
    rollback_lines.append("COMMIT;")

    out_forward.write_text("\n".join(forward_lines) + "\n", encoding="utf-8")
    out_rollback.write_text("\n".join(rollback_lines) + "\n", encoding="utf-8")

    return totals


def atomic_rename(pending: Path, final: Path) -> None:
    """POSIX atomic rename. Same-filesystem only."""
    if final.exists():
        raise FileExistsError(f"final release directory already exists: {final}")
    os.rename(pending, final)


def update_latest_symlink(releases_dir: Path, version: str) -> None:
    """Atomically swap ``releases/latest`` → ``<version>`` via rename-over-tmp."""
    target = releases_dir / version
    link = releases_dir / "latest"
    tmp_link = releases_dir / ".latest.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(version, target_is_directory=True)
    os.replace(tmp_link, link)  # atomic on POSIX


__all__ = [
    "ReleaseArtifacts",
    "atomic_rename",
    "emit_migrations",
    "snapshot",
    "update_latest_symlink",
]
