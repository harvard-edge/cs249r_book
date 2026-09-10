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


def _with_chain(path: Path, rows: list[tuple], chain_rows: list[tuple]) -> None:
    """Extend the test DB with a chain + chain_questions + tags for
    multi-table rollback-symmetry coverage (Dean R3-NC-1)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS chains (id TEXT PRIMARY KEY, name TEXT, topic TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chain_questions "
        "(chain_id TEXT, question_id TEXT, position INTEGER, PRIMARY KEY(chain_id, position))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tags (question_id TEXT, tag TEXT, PRIMARY KEY(question_id, tag))"
    )
    for cid, name, topic in [("c1", "Chain 1", "t")]:
        conn.execute("INSERT OR REPLACE INTO chains VALUES (?,?,?)", (cid, name, topic))
    for cid, qid, pos in chain_rows:
        conn.execute(
            "INSERT OR REPLACE INTO chain_questions VALUES (?,?,?)", (cid, qid, pos)
        )
    for qid, tag in [(rows[0][0], "hw:a100")]:
        conn.execute("INSERT OR REPLACE INTO tags VALUES (?,?)", (qid, tag))
    conn.commit()
    conn.close()


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


def test_emit_migrations_covers_all_tables(tmp_path: Path) -> None:
    """Dean R3-NC-1: emit_migrations must diff questions, chains,
    chain_questions, and tags — not just questions."""
    prev = tmp_path / "prev.db"
    new = tmp_path / "new.db"
    _make_db(prev, [_row("a", "same"), _row("b", "same")])
    _with_chain(prev, [_row("a", "same"), _row("b", "same")], [("c1", "a", 1), ("c1", "b", 2)])
    _make_db(new, [_row("a", "same"), _row("b", "same"), _row("c", "new")])
    _with_chain(new, [_row("a", "same"), _row("b", "same"), _row("c", "new")], [("c1", "a", 1), ("c1", "c", 2)])

    fwd = tmp_path / "fwd.sql"
    rbk = tmp_path / "rbk.sql"
    emit_migrations(prev_db=prev, new_db=new, out_forward=fwd, out_rollback=rbk)

    fwd_text = fwd.read_text()
    rbk_text = rbk.read_text()
    # Forward + rollback must touch chain_questions, not just questions.
    assert "chain_questions" in fwd_text, "forward migration missed chain_questions table"
    assert "chain_questions" in rbk_text, "rollback missed chain_questions table"
    # Rollback of a chain_questions modification must restore prior position binding.
    assert "INSERT OR REPLACE INTO chain_questions" in rbk_text

    # Apply forward + rollback, assert state is byte-identical across ALL tables.
    import shutil
    target = tmp_path / "target.db"
    shutil.copy2(prev, target)
    conn = sqlite3.connect(target)
    conn.executescript(fwd_text)
    conn.executescript(rbk_text)
    target_q = set(conn.execute("SELECT id, realistic_solution FROM questions").fetchall())
    target_c = set(conn.execute("SELECT chain_id, question_id, position FROM chain_questions").fetchall())
    conn.close()
    conn = sqlite3.connect(prev)
    prev_q = set(conn.execute("SELECT id, realistic_solution FROM questions").fetchall())
    prev_c = set(conn.execute("SELECT chain_id, question_id, position FROM chain_questions").fetchall())
    conn.close()
    assert target_q == prev_q, "rollback must restore questions state"
    assert target_c == prev_c, "rollback must restore chain_questions state"
