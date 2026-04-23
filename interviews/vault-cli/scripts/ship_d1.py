#!/usr/bin/env python3
"""Ship the current vault.db to the live Cloudflare D1 database.

Generates a full-reload SQL script (DROP + CREATE + INSERT) from a fresh
vault.db build, then applies it via `wrangler d1 execute --remote`.

Usage:
    # Fresh build + push to production D1:
    python3 interviews/vault-cli/scripts/ship_d1.py

    # Dry-run: write SQL to /tmp/d1_cutover.sql, don't apply:
    python3 interviews/vault-cli/scripts/ship_d1.py --dry-run

    # Stage instead of production:
    python3 interviews/vault-cli/scripts/ship_d1.py --env staging

Requires:
    - `vault` CLI installed (`pip install -e interviews/vault-cli/`)
    - `wrangler` CLI authenticated (`wrangler whoami`)
"""

from __future__ import annotations

import argparse
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO / "interviews" / "vault"
VAULT_DB = VAULT_DIR / "vault.db"
WORKER_DIR = REPO / "interviews" / "staffml-vault-worker"
D1_SCHEMA = REPO / "interviews" / "vault-cli" / "scripts" / "d1-schema.sql"


def _sql_lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def generate_sql(vault_db: Path, d1_schema: Path, out: Path) -> None:
    parts = []
    parts.append("-- D1 full-reload migration — DROP + CREATE + INSERT")
    parts.append(f"-- Generated from {vault_db.name} (byte-for-byte reproducible)")
    parts.append("PRAGMA foreign_keys = OFF;")
    for tbl in ["questions_fts", "chain_questions", "chains", "tags", "release_metadata", "questions"]:
        parts.append(f"DROP TABLE IF EXISTS {tbl};")
    for trig in ["questions_ai", "questions_ad", "questions_au"]:
        parts.append(f"DROP TRIGGER IF EXISTS {trig};")
    parts.append("")
    parts.append(d1_schema.read_text())
    parts.append("PRAGMA foreign_keys = ON;")

    conn = sqlite3.connect(vault_db)
    cur = conn.cursor()

    # Parent tables first so FKs don't bounce the child inserts.
    parts.append("\n-- chains (parent of chain_questions)")
    for row in cur.execute("SELECT id, name, topic FROM chains"):
        parts.append(f"INSERT INTO chains VALUES ({','.join(_sql_lit(v) for v in row)});")

    parts.append("\n-- questions")
    for row in cur.execute("""
        SELECT id, title, topic, track, level, zone, competency_area, bloom_level, phase,
               status, scenario, common_mistake, realistic_solution, napkin_math,
               provenance, created_at, last_modified,
               human_review_status, human_review_by, human_review_date,
               file_path, content_hash, authors_json
        FROM questions
    """):
        parts.append(f"INSERT INTO questions VALUES ({','.join(_sql_lit(v) for v in row)});")

    parts.append("\n-- chain_questions")
    for row in cur.execute("SELECT chain_id, question_id, position FROM chain_questions"):
        parts.append(f"INSERT INTO chain_questions VALUES ({','.join(_sql_lit(v) for v in row)});")

    parts.append("\n-- tags")
    for row in cur.execute("SELECT question_id, tag FROM tags"):
        parts.append(f"INSERT INTO tags VALUES ({','.join(_sql_lit(v) for v in row)});")

    parts.append("\n-- release_metadata")
    for row in cur.execute("SELECT key, value FROM release_metadata"):
        parts.append(f"INSERT INTO release_metadata VALUES ({','.join(_sql_lit(v) for v in row)});")

    conn.close()
    out.write_text("\n".join(parts) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ship vault.db to Cloudflare D1.")
    ap.add_argument("--env", choices=["production", "staging"], default="production",
                    help="Target D1 environment (default: production).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write SQL to /tmp/d1_cutover.sql but do not apply.")
    ap.add_argument("--skip-build", action="store_true",
                    help="Assume vault.db is already up to date.")
    args = ap.parse_args()

    if not args.skip_build:
        print("==> vault build", flush=True)
        subprocess.check_call(
            ["vault", "build", "--vault-dir", str(VAULT_DIR), "--release-id", f"ship-{args.env}"],
            cwd=REPO,
        )

    sql_path = Path("/tmp/d1_cutover.sql")
    print(f"==> generating {sql_path} from {VAULT_DB.name}", flush=True)
    generate_sql(VAULT_DB, D1_SCHEMA, sql_path)
    size_mb = sql_path.stat().st_size / 1024 / 1024
    with sql_path.open() as fh:
        line_count = sum(1 for _ in fh)
    print(f"    wrote {line_count} lines ({size_mb:.1f} MiB)", flush=True)

    if args.dry_run:
        print("==> dry-run; skipping wrangler d1 execute", flush=True)
        return 0

    env_flag = "" if args.env == "production" else f"--env {args.env}"
    cmd = f"npx wrangler d1 execute staffml-vault --remote {env_flag} --file={sql_path}".strip()
    print(f"==> {cmd}", flush=True)
    subprocess.check_call(shlex.split(cmd), cwd=WORKER_DIR)
    print("==> done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
