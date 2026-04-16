"""Tests for release pipeline primitives."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from vault_cli.release import emit_migrations, snapshot


def _make_db(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE questions (
            id TEXT PRIMARY KEY, title TEXT, topic TEXT, track TEXT, level TEXT,
            zone TEXT, status TEXT, scenario TEXT, common_mistake TEXT,
            realistic_solution TEXT, napkin_math TEXT, deep_dive_title TEXT,
            deep_dive_url TEXT, provenance TEXT, created_at TEXT, last_modified TEXT,
            file_path TEXT, content_hash TEXT, authors_json TEXT)
    """)
    conn.execute("CREATE TABLE release_metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.execute("INSERT INTO release_metadata VALUES ('release_hash', 'test')")
    conn.execute("INSERT INTO release_metadata VALUES ('policy_version', '1')")
    conn.execute("INSERT INTO release_metadata VALUES ('schema_version', '1')")
    conn.execute("INSERT INTO release_metadata VALUES ('published_count', ?)",
                 (str(len(rows)),))
    conn.commit()
    conn.close()


def _row(qid: str, solution: str) -> tuple:
    return (qid, "T", "topic", "cloud", "l4", "recall", "published", "scn",
            None, solution, None, None, None, "human", None, None,
            f"/tmp/{qid}.yaml", "hash-" + qid, None)


def test_snapshot_copies_db_and_writes_release_json(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    _make_db(db, [_row("a", "answer-a")])
    releases = tmp_path / "releases"
    artifacts = snapshot(db, releases, "1.0.0")
    assert artifacts.directory == releases / ".pending-1.0.0"
    assert artifacts.vault_db.exists()
    assert artifacts.release_json.exists()


def test_migrations_emit_added_modified_removed(tmp_path: Path) -> None:
    prev = tmp_path / "prev.db"
    new = tmp_path / "new.db"
    _make_db(prev, [_row("a", "old"), _row("b", "keep")])
    _make_db(new, [_row("a", "updated"), _row("c", "new")])
    fwd = tmp_path / "fwd.sql"
    rbk = tmp_path / "rbk.sql"
    stats = emit_migrations(prev_db=prev, new_db=new, out_forward=fwd, out_rollback=rbk)
    assert stats == {"added": 1, "removed": 1, "modified": 1}
    assert fwd.exists() and rbk.exists()
    # Rollback must embed prior-row body for DELETEs (fixes C-1).
    assert "INSERT OR REPLACE INTO questions" in rbk.read_text()
    assert "old" in rbk.read_text()  # prior body of 'a' embedded for rollback


def test_rollback_symmetry_property(tmp_path: Path) -> None:
    """Forward-then-rollback must return dump identical to pre-migration state."""
    prev = tmp_path / "prev.db"
    new = tmp_path / "new.db"
    _make_db(prev, [_row("a", "old"), _row("b", "keep")])
    _make_db(new, [_row("a", "updated"), _row("c", "new")])
    fwd = tmp_path / "fwd.sql"
    rbk = tmp_path / "rbk.sql"
    emit_migrations(prev_db=prev, new_db=new, out_forward=fwd, out_rollback=rbk)

    # Simulate deploy: apply forward to a copy of prev, then rollback, compare to prev.
    import shutil
    target = tmp_path / "target.db"
    shutil.copy2(prev, target)
    conn = sqlite3.connect(target)
    conn.executescript(fwd.read_text())
    conn.executescript(rbk.read_text())
    # Dump rows from target; compare to prev.
    target_rows = set(conn.execute("SELECT id, realistic_solution FROM questions").fetchall())
    conn.close()
    conn = sqlite3.connect(prev)
    prev_rows = set(conn.execute("SELECT id, realistic_solution FROM questions").fetchall())
    conn.close()
    assert target_rows == prev_rows, "rollback must restore pre-migration state"
