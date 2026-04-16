"""YAML → SQLite compiler.

Produces ``vault.db`` as a build artifact. Consumed by:
- ``vault export paper`` (SQL → LaTeX macros),
- D1 migration emitter (SQL → UPSERT deltas),
- ``vault serve`` via Datasette for ad-hoc exploration.

The SQLite file is never hashed directly (ARCHITECTURE.md §3.5 — not
byte-reproducible). ``release_hash`` is computed over YAML inputs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vault_cli.hashing import content_hash, hash_of_canonical_yaml, release_hash
from vault_cli.loader import LoadedQuestion
from vault_cli.policy import filter_questions, load_policy, policy_version
from vault_cli.yaml_io import load_file

DDL = """
CREATE TABLE questions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  topic TEXT NOT NULL,
  track TEXT NOT NULL,
  level TEXT NOT NULL,
  zone TEXT NOT NULL,
  status TEXT NOT NULL,
  scenario TEXT NOT NULL,
  common_mistake TEXT,
  realistic_solution TEXT NOT NULL,
  napkin_math TEXT,
  deep_dive_title TEXT,
  deep_dive_url TEXT,
  provenance TEXT NOT NULL,
  created_at TEXT,
  last_modified TEXT,
  file_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  authors_json TEXT
);

CREATE INDEX idx_questions_topic ON questions(topic);
CREATE INDEX idx_questions_track_level ON questions(track, level);
CREATE INDEX idx_questions_zone ON questions(zone);
CREATE INDEX idx_questions_status ON questions(status);

CREATE TABLE chains (
  id TEXT PRIMARY KEY,
  name TEXT,
  topic TEXT
);

CREATE TABLE chain_questions (
  chain_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  PRIMARY KEY (chain_id, position),
  FOREIGN KEY (chain_id) REFERENCES chains(id),
  FOREIGN KEY (question_id) REFERENCES questions(id)
);

CREATE TABLE tags (
  question_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES questions(id)
);

CREATE TABLE release_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Full-text search (B.5). Content-table pattern keeps the FTS index in sync
-- with INSERTs/UPDATEs/DELETEs via triggers. Cold-start fingerprint check
-- includes triggers — see staffml-vault-worker/src/index.ts.
CREATE VIRTUAL TABLE questions_fts USING fts5(
  title, scenario, realistic_solution,
  content='questions', content_rowid='rowid'
);

CREATE TRIGGER questions_ai AFTER INSERT ON questions BEGIN
  INSERT INTO questions_fts(rowid, title, scenario, realistic_solution)
  VALUES (new.rowid, new.title, new.scenario, new.realistic_solution);
END;

CREATE TRIGGER questions_ad AFTER DELETE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, title, scenario, realistic_solution)
  VALUES('delete', old.rowid, old.title, old.scenario, old.realistic_solution);
END;

CREATE TRIGGER questions_au AFTER UPDATE ON questions BEGIN
  INSERT INTO questions_fts(questions_fts, rowid, title, scenario, realistic_solution)
  VALUES('delete', old.rowid, old.title, old.scenario, old.realistic_solution);
  INSERT INTO questions_fts(rowid, title, scenario, realistic_solution)
  VALUES (new.rowid, new.title, new.scenario, new.realistic_solution);
END;
"""


def _compile_questions(
    conn: sqlite3.Connection,
    loaded: Iterable[LoadedQuestion],
) -> list[tuple[str, str]]:
    """Insert rows; return list of (id, content_hash) for Merkle construction."""
    leaves: list[tuple[str, str]] = []
    cur = conn.cursor()
    for lq in loaded:
        q = lq.question
        payload = q.model_dump(mode="json", exclude_none=False)
        ch = content_hash(payload)
        leaves.append((q.id, ch))

        dd = q.details.deep_dive
        cur.execute(
            """
            INSERT INTO questions (
              id, title, topic, track, level, zone, status,
              scenario, common_mistake, realistic_solution, napkin_math,
              deep_dive_title, deep_dive_url, provenance,
              created_at, last_modified, file_path, content_hash, authors_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                q.id,
                q.title,
                q.topic,
                lq.classification.track.value,
                lq.classification.level.value,
                lq.classification.zone.value,
                q.status.value,
                q.scenario,
                q.details.common_mistake,
                q.details.realistic_solution,
                q.details.napkin_math,
                dd.title if dd else None,
                dd.url if dd else None,
                q.provenance.value,
                q.created_at.isoformat() if q.created_at else None,
                q.last_modified.isoformat() if q.last_modified else None,
                str(lq.path),
                ch,
                json.dumps(q.authors) if q.authors else None,
            ),
        )
        if q.chain is not None:
            cur.execute(
                "INSERT OR IGNORE INTO chain_questions(chain_id, question_id, position) VALUES (?,?,?)",
                (q.chain.id, q.id, q.chain.position),
            )
        for tag in q.tags or []:
            cur.execute("INSERT INTO tags(question_id, tag) VALUES (?, ?)", (q.id, tag))
    conn.commit()
    return leaves


def build(
    *,
    vault_dir: Path,
    loaded: list[LoadedQuestion],
    output: Path,
    release_id: str = "dev",
) -> dict[str, Any]:
    """Compile loaded questions to a SQLite file and return release metadata."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    policy = load_policy(vault_dir / "release-policy.yaml")
    published = filter_questions((lq.question.model_dump(mode="json") for lq in loaded), policy)
    published_ids = {q["id"] for q in published}
    released = [lq for lq in loaded if lq.id in published_ids]

    conn = sqlite3.connect(output)
    try:
        conn.executescript(DDL)
        leaves = _compile_questions(conn, released)

        taxonomy_path = vault_dir / "taxonomy.yaml"
        chains_path = vault_dir / "chains.yaml"
        zones_path = vault_dir / "zones.yaml"
        taxonomy_hash = hash_of_canonical_yaml(load_file(taxonomy_path)) if taxonomy_path.exists() else hash_of_canonical_yaml({})
        chains_hash = hash_of_canonical_yaml(load_file(chains_path)) if chains_path.exists() else hash_of_canonical_yaml({})
        zones_hash = hash_of_canonical_yaml(load_file(zones_path)) if zones_path.exists() else hash_of_canonical_yaml({})
        policy_hash = hash_of_canonical_yaml(policy)

        rhash = release_hash(
            per_question=leaves,
            taxonomy_hash=taxonomy_hash,
            chains_hash=chains_hash,
            zones_hash=zones_hash,
            policy_hash=policy_hash,
        )

        meta = {
            "release_id": release_id,
            "release_hash": rhash,
            "policy_version": str(policy_version(policy)),
            "schema_version": "1",
            "published_count": str(len(released)),
        }
        cur = conn.cursor()
        for k, v in meta.items():
            cur.execute("INSERT INTO release_metadata(key, value) VALUES (?, ?)", (k, v))
        conn.commit()
    finally:
        conn.close()

    return {
        "output": str(output),
        "release_id": release_id,
        "release_hash": rhash,
        "published_count": len(released),
        "policy_version": policy_version(policy),
    }


__all__ = ["DDL", "build"]
