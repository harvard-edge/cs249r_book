"""Tests for the newer subcommands: doctor, diff, stats, codegen.

These exercise the command surfaces end-to-end via Typer's CliRunner so a
stale --json schema, exit-code drift, or a regression in one of the
subchecks is caught in CI.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer
from typer.testing import CliRunner

from vault_cli.commands import codegen as codegen_mod
from vault_cli.commands import diff_cmd as diff_mod
from vault_cli.commands import doctor as doctor_mod
from vault_cli.commands import stats as stats_mod
from vault_cli.exit_codes import ExitCode


def _make_vault(tmp: Path, questions: list[tuple[str, str]]) -> Path:
    """Minimal vault.db with a questions table populated from (id, title) pairs."""
    vault_dir = tmp / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    db = vault_dir / "vault.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE questions (
            id TEXT PRIMARY KEY, title TEXT, topic TEXT, track TEXT, level TEXT,
            zone TEXT, status TEXT, scenario TEXT, common_mistake TEXT,
            realistic_solution TEXT, napkin_math TEXT, deep_dive_title TEXT,
            deep_dive_url TEXT, provenance TEXT, created_at TEXT, last_modified TEXT,
            file_path TEXT, content_hash TEXT, authors_json TEXT);
        CREATE TABLE chains (id TEXT PRIMARY KEY, name TEXT, topic TEXT);
        CREATE TABLE chain_questions (chain_id TEXT, question_id TEXT, position INTEGER,
            PRIMARY KEY(chain_id, position));
        CREATE TABLE tags (question_id TEXT, tag TEXT, PRIMARY KEY(question_id, tag));
        CREATE TABLE release_metadata (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    for qid, title in questions:
        conn.execute(
            """INSERT INTO questions VALUES
               (?, ?, 't', 'cloud', 'l1', 'recall', 'published', 'scn',
                NULL, 'soln', NULL, NULL, NULL, 'imported',
                NULL, NULL, '/tmp/x.yaml', 'hash-' + ?, NULL)""".replace("'hash-' + ?", "'h-'||?"),
            (qid, title, qid),
        )
    meta = {
        "release_id": "0.1.0", "release_hash": "a" * 64,
        "schema_version": "1", "policy_version": "1",
        "published_count": str(len(questions)),
    }
    for k, v in meta.items():
        conn.execute("INSERT INTO release_metadata VALUES (?, ?)", (k, v))
    conn.commit()
    conn.close()
    return vault_dir


def _app(register_fn) -> typer.Typer:
    """Build a multi-command Typer app so CliRunner invokes
    `<app> <subcommand> [args]` rather than single-command mode.
    Typer promotes a single-command app to root; we add a no-op callback
    plus a second dummy command to keep multi-command behavior stable.
    """
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Root — tests use subcommand invocation."""

    @app.command("_noop")
    def _noop() -> None:
        """Keeps Typer in multi-command mode."""

    register_fn(app)
    return app


def test_stats_json_shape(tmp_path: Path) -> None:
    _make_vault(tmp_path, [("global-0000", "A"), ("global-0001", "B")])
    app = _app(stats_mod.register)
    r = CliRunner().invoke(app, ["stats", "--vault-db", str(tmp_path / "vault" / "vault.db"), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["ok"] is True
    assert data["data"]["total"] == 2
    assert data["data"]["release_id"] == "0.1.0"


def test_stats_prometheus_format(tmp_path: Path) -> None:
    _make_vault(tmp_path, [("global-0000", "A")])
    app = _app(stats_mod.register)
    r = CliRunner().invoke(app, [
        "stats", "--vault-db", str(tmp_path / "vault" / "vault.db"), "--format-prometheus",
    ])
    assert r.exit_code == 0
    assert "vault_questions_total" in r.stdout
    assert "vault_questions_by_track" in r.stdout


def test_diff_cosmetic_semantic_structural(tmp_path: Path) -> None:
    releases = tmp_path / "releases"
    _make_vault(tmp_path / "v1", [("a", "A"), ("b", "B")])
    _make_vault(tmp_path / "v2", [("a", "A"), ("b", "B"), ("c", "C")])
    (releases / "v1").mkdir(parents=True)
    (releases / "v2").mkdir(parents=True)
    (tmp_path / "v1" / "vault" / "vault.db").rename(releases / "v1" / "vault.db")
    (tmp_path / "v2" / "vault" / "vault.db").rename(releases / "v2" / "vault.db")

    app = _app(diff_mod.register)
    r = CliRunner().invoke(app, [
        "diff", "v1", "v2", "--releases-dir", str(releases), "--json",
    ])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert len(data["data"]["added"]) == 1
    assert data["data"]["added"][0]["id"] == "c"
    assert len(data["data"]["removed"]) == 0


def test_doctor_skip_when_no_vault(tmp_path: Path) -> None:
    """Doctor must not crash on a bare vault_dir; subchecks should skip gracefully."""
    empty_vault = tmp_path / "empty"
    empty_vault.mkdir()
    app = _app(doctor_mod.register)
    r = CliRunner().invoke(app, [
        "doctor", "--vault-dir", str(empty_vault), "--json", "--check", "release-integrity",
    ])
    # release-integrity returns skip when no releases/ dir — not a failure.
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert data["data"]["checks"][0]["check"] == "release-integrity"
    assert data["data"]["checks"][0]["status"] in {"skip", "warn"}


def test_doctor_unknown_check_usage_error(tmp_path: Path) -> None:
    app = _app(doctor_mod.register)
    r = CliRunner().invoke(app, ["doctor", "--check", "no-such-check"])
    assert r.exit_code == ExitCode.USAGE_ERROR


def test_codegen_check_records_or_verifies_baseline(tmp_path: Path) -> None:
    app = _app(codegen_mod.register)
    r = CliRunner().invoke(app, ["codegen", "--check"])
    # Should either record baseline (first run) or verify clean. Never fail
    # cleanly when the 3 artifacts exist and are non-empty in this repo.
    assert r.exit_code == 0, r.output
