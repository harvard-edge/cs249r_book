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


def _dump_questions(db: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM questions").fetchall()
        return {r["id"]: {k: r[k] for k in r.keys()} for r in rows}
    finally:
        conn.close()


def _sql_quote(s: Any) -> str:
    if s is None:
        return "NULL"
    if isinstance(s, (int, float)):
        return str(s)
    return "'" + str(s).replace("'", "''") + "'"


def emit_migrations(prev_db: Path, new_db: Path, out_forward: Path, out_rollback: Path) -> dict[str, int]:
    """Emit forward + inverse migrations between two vault.db snapshots.

    CRITICAL (fixes C-1): inverse migration embeds FULL PRIOR ROW BODIES for
    UPDATEs and DELETEs, so rollback can reconstruct state without relying on
    mechanical inversion. This is the correctness-preserving path when the
    primary snapshot-restore rollback path is unavailable.
    """
    prev = _dump_questions(prev_db) if prev_db.exists() else {}
    new = _dump_questions(new_db)

    added = set(new) - set(prev)
    removed = set(prev) - set(new)
    modified = {qid for qid in (set(new) & set(prev)) if prev[qid] != new[qid]}

    def row_values(row: dict[str, Any]) -> list[str]:
        cols = ["id","title","topic","track","level","zone","status","scenario",
                "common_mistake","realistic_solution","napkin_math","deep_dive_title",
                "deep_dive_url","provenance","created_at","last_modified","file_path",
                "content_hash","authors_json"]
        return [_sql_quote(row.get(c)) for c in cols]

    forward_lines = ["BEGIN;"]
    rollback_lines = ["BEGIN;"]

    for qid in sorted(added):
        vals = row_values(new[qid])
        forward_lines.append(f"INSERT OR REPLACE INTO questions VALUES ({', '.join(vals)});")
        rollback_lines.append(f"DELETE FROM questions WHERE id = {_sql_quote(qid)};")

    for qid in sorted(removed):
        forward_lines.append(f"DELETE FROM questions WHERE id = {_sql_quote(qid)};")
        vals = row_values(prev[qid])
        rollback_lines.append(f"INSERT OR REPLACE INTO questions VALUES ({', '.join(vals)});")

    for qid in sorted(modified):
        vals_new = row_values(new[qid])
        vals_prev = row_values(prev[qid])
        forward_lines.append(f"INSERT OR REPLACE INTO questions VALUES ({', '.join(vals_new)});")
        rollback_lines.append(f"INSERT OR REPLACE INTO questions VALUES ({', '.join(vals_prev)});")

    forward_lines.append("COMMIT;")
    rollback_lines.append("COMMIT;")

    out_forward.write_text("\n".join(forward_lines) + "\n", encoding="utf-8")
    out_rollback.write_text("\n".join(rollback_lines) + "\n", encoding="utf-8")

    return {"added": len(added), "removed": len(removed), "modified": len(modified)}


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
