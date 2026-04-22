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
  competency_area TEXT,
  bloom_level TEXT,
  phase TEXT,
  status TEXT NOT NULL,
  scenario TEXT NOT NULL,
  common_mistake TEXT,
  realistic_solution TEXT NOT NULL,
  napkin_math TEXT,
  provenance TEXT NOT NULL,
  created_at TEXT,
  last_modified TEXT,
  human_review_status TEXT,
  human_review_by TEXT,
  human_review_date TEXT,
  file_path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  authors_json TEXT
);

CREATE INDEX idx_questions_topic ON questions(topic);
CREATE INDEX idx_questions_track_level ON questions(track, level);
CREATE INDEX idx_questions_zone ON questions(zone);
CREATE INDEX idx_questions_status ON questions(status);
CREATE INDEX idx_questions_human_review ON questions(human_review_status);

CREATE TABLE chains (
  id TEXT PRIMARY KEY,
  name TEXT,
  topic TEXT
);

-- v1.0: a question may belong to multiple chains; PK is (chain_id, question_id)
-- rather than (chain_id, position) to allow non-contiguous/0-indexed positions.
CREATE TABLE chain_questions (
  chain_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  PRIMARY KEY (chain_id, question_id),
  FOREIGN KEY (chain_id) REFERENCES chains(id),
  FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_chain_questions_qid ON chain_questions(question_id);

CREATE TABLE tags (
  question_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES questions(id)
);

CREATE TABLE taxonomy (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  area TEXT NOT NULL,
  prerequisites_json TEXT,
  tracks_json TEXT,
  question_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_taxonomy_area ON taxonomy(area);

CREATE TABLE taxonomy_edges (
  source TEXT NOT NULL,
  target TEXT NOT NULL,
  PRIMARY KEY (source, target),
  FOREIGN KEY (source) REFERENCES taxonomy(id),
  FOREIGN KEY (target) REFERENCES taxonomy(id)
);

CREATE TABLE zones (
  id TEXT PRIMARY KEY,
  description TEXT,
  skills_json TEXT,
  levels_json TEXT
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

        # v1.0: classification is on the question itself, not a side-channel.
        # DeepDive/deep_dive fields were retired in favor of details.resources[].
        hr = q.human_reviewed
        cur.execute(
            """
            INSERT INTO questions (
              id, title, topic, track, level, zone, competency_area, bloom_level, phase,
              status, scenario, common_mistake, realistic_solution, napkin_math,
              provenance, created_at, last_modified,
              human_review_status, human_review_by, human_review_date,
              file_path, content_hash, authors_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                q.id,
                q.title,
                q.topic,
                q.track,
                q.level,
                q.zone,
                q.competency_area,
                q.bloom_level,
                q.phase,
                q.status,
                q.scenario,
                q.details.common_mistake,
                q.details.realistic_solution,
                q.details.napkin_math,
                q.provenance,
                str(q.created_at) if q.created_at else None,
                str(q.last_modified) if q.last_modified else None,
                hr.status if hr else None,
                hr.by if hr else None,
                str(hr.date) if hr and hr.date else None,
                str(lq.path),
                ch,
                json.dumps(q.authors) if q.authors else None,
            ),
        )
        # v1.0 plural chains: a question can belong to multiple chains.
        for c in q.chains or []:
            cur.execute(
                "INSERT OR IGNORE INTO chain_questions(chain_id, question_id, position) VALUES (?,?,?)",
                (c.id, q.id, c.position),
            )
        for tag in q.tags or []:
            cur.execute("INSERT INTO tags(question_id, tag) VALUES (?, ?)", (q.id, tag))
    conn.commit()
    return leaves


def _populate_taxonomy(conn: sqlite3.Connection, data_dir: Path) -> None:
    """Load taxonomy.json into the taxonomy + taxonomy_edges tables."""
    tax_path = data_dir / "taxonomy.json"
    if not tax_path.exists():
        return
    data = json.loads(tax_path.read_text(encoding="utf-8"))
    concepts = data.get("concepts", [])
    cur = conn.cursor()
    for c in concepts:
        cur.execute(
            "INSERT OR REPLACE INTO taxonomy(id, name, description, area, "
            "prerequisites_json, tracks_json, question_count) VALUES (?,?,?,?,?,?,?)",
            (
                c["id"],
                c["name"],
                c.get("description", ""),
                c["area"],
                json.dumps(c.get("prerequisites", [])),
                json.dumps(c.get("tracks", [])),
                c.get("question_count", 0),
            ),
        )
        for prereq in c.get("prerequisites", []):
            cur.execute(
                "INSERT OR IGNORE INTO taxonomy_edges(source, target) VALUES (?,?)",
                (prereq, c["id"]),
            )
    conn.commit()


def _populate_chains(conn: sqlite3.Connection, vault_dir: Path) -> None:
    """Load chains.json into the `chains` table.

    The compiler previously left this table empty, which made
    chain_questions FK constraints fail on any engine that enforces
    them (e.g., Cloudflare D1).
    """
    chains_path = vault_dir / "chains.json"
    if not chains_path.exists():
        return
    data = json.loads(chains_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return
    cur = conn.cursor()
    for c in data:
        if not isinstance(c, dict):
            continue
        cid = c.get("chain_id") or c.get("id")
        if not cid:
            continue
        cur.execute(
            "INSERT OR REPLACE INTO chains(id, name, topic) VALUES (?, ?, ?)",
            (cid, c.get("name"), c.get("topic")),
        )
    conn.commit()


def _populate_zones(conn: sqlite3.Connection, data_dir: Path) -> None:
    """Load zones.json into the zones table."""
    zones_path = data_dir / "zones.json"
    if not zones_path.exists():
        return
    data = json.loads(zones_path.read_text(encoding="utf-8"))
    zones = data.get("zones", {})
    cur = conn.cursor()
    for zone_id, info in zones.items():
        cur.execute(
            "INSERT OR REPLACE INTO zones(id, description, skills_json, levels_json) "
            "VALUES (?,?,?,?)",
            (
                zone_id,
                info.get("description", ""),
                json.dumps(info.get("skills", [])),
                json.dumps(info.get("levels", [])),
            ),
        )
    conn.commit()


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

        # ── Populate taxonomy + zones tables from StaffML JSON data ──
        staffml_data = (vault_dir / ".." / "staffml" / "src" / "data").resolve()
        _populate_taxonomy(conn, staffml_data)
        _populate_zones(conn, staffml_data)

        # ── Populate chains table from chains.json ──
        # Without this the chains table is empty and FK enforcement (D1)
        # rejects chain_questions inserts pointing at unknown chain IDs.
        _populate_chains(conn, vault_dir)

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

        # Gemini R5-C-3 + Chip R7-H-2: schema_fingerprint is computed over
        # USER-AUTHORED DDL only. Previously hashed all of sqlite_master
        # including FTS5 auto-generated shadow tables (questions_fts_data,
        # _idx, _docsize, _content, _config) whose DDL text varies across
        # SQLite versions. Host Python's SQLite and Cloudflare D1's SQLite
        # are NOT guaranteed to agree on shadow-table DDL — fingerprint
        # mismatches permanently, worker pinned to degraded mode.
        # Fix: filter by name pattern to user-authored objects.
        import hashlib
        import re
        ddl_rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type IN ('table','index','trigger','view') "
            "AND name NOT LIKE 'sqlite_%' "
            # Exclude FTS5 shadow tables (SQLite-version-dependent DDL).
            "AND name NOT IN ('questions_fts_data','questions_fts_idx',"
            "                 'questions_fts_docsize','questions_fts_content',"
            "                 'questions_fts_config') "
            "AND name NOT LIKE '_cf_%' "
            "AND name NOT LIKE 'd1_%' "
            "ORDER BY name"
        ).fetchall()
        ddl_text = "\n".join(
            re.sub(r"\s+", " ", (sql or "")).strip()
            for (_name, sql) in ddl_rows
            if sql
        )
        schema_fingerprint = hashlib.sha256(ddl_text.encode("utf-8")).hexdigest()

        meta = {
            "release_id": release_id,
            "release_hash": rhash,
            "policy_version": str(policy_version(policy)),
            "schema_version": "1",
            "published_count": str(len(released)),
            "schema_fingerprint": schema_fingerprint,
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
