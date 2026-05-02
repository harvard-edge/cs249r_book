#!/usr/bin/env python3
"""One-shot split of the monolithic corpus.json into per-question YAML files.

This is the Phase-1 foundation operation. After this runs:
- Every question from ``vault/corpus.json`` exists as a YAML file at
  ``vault/questions/<track>/<level>/<zone>/<id>.yaml``.
- The content_hash of each YAML (via the canonical hasher) matches the
  content_hash computable from the original corpus.json record.
- ``id-registry.yaml`` contains one append-only entry per question.

Reruns are idempotent by default (skips existing files). Use ``--force`` to
overwrite.

Referenced from ARCHITECTURE.md §14 Phase 1; REVIEWS.md §11.1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure local imports work when script run outside of installed env.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "interviews" / "vault-cli" / "src"))

from vault_cli.yaml_io import dump_str  # noqa: E402

INTERVIEWS = _REPO / "interviews"
VAULT = INTERVIEWS / "vault"
CORPUS_JSON = VAULT / "corpus.json"
QUESTIONS_ROOT = VAULT / "questions"
REGISTRY_PATH = VAULT / "id-registry.yaml"


# Slugify for filenames + topic paths.
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    slug = _SLUG_RE.sub("-", s.lower()).strip("-")
    return slug or "untitled"


def _coerce_status(raw: str | None) -> str:
    if raw in {"published", "draft", "deprecated"}:
        return raw
    if raw in {"approved", "review"}:  # legacy values
        return "draft"
    return "draft"


def _coerce_level(raw: str | None) -> str:
    if raw is None:
        return "l1"
    lvl = raw.lower()
    if lvl in {"l1", "l2", "l3", "l4", "l5", "l6"}:
        return lvl
    return "l1"


def _coerce_zone(raw: str | None) -> str:
    known = {"recall", "fluency", "implement", "specification",
             "analyze", "diagnosis", "design", "evaluation"}
    if raw and raw.lower() in known:
        return raw.lower()
    return "recall"


def _coerce_track(raw: str | None) -> str:
    known = {"cloud", "edge", "mobile", "tinyml", "global"}
    if raw and raw.lower() in known:
        return raw.lower()
    return "global"


def _infer_provenance(record: dict[str, Any]) -> str:
    """Phase-1 conservative default: all legacy corpus entries tagged 'imported'.

    Provenance backfill is follow-up work: maintainers retag as ``human`` or
    ``llm-then-human-edited`` after review. Tagging as ``imported`` is honest;
    tagging as ``human`` would be a lie.
    """
    return "imported"


def _extract_chain(record: dict[str, Any]) -> dict[str, Any] | None:
    """Legacy ``chain_positions`` is a {chain_id: 0-indexed-position} dict.

    New schema uses 1-indexed positions, so we add 1 on the way in.
    """
    chain_ids = record.get("chain_ids") or []
    positions = record.get("chain_positions") or {}
    if not chain_ids:
        return None
    cid = chain_ids[0]
    if isinstance(positions, dict):
        pos = int(positions.get(cid, 0)) + 1
    elif isinstance(positions, list):
        pos = int(positions[0]) + 1 if positions else 1
    else:
        pos = 1
    return {"id": cid, "position": max(1, pos)}


def _extract_details(record: dict[str, Any]) -> dict[str, Any]:
    d = record.get("details") or {}
    if not isinstance(d, dict):
        d = {}
    out: dict[str, Any] = {
        "realistic_solution": d.get("realistic_solution") or d.get("solution") or "",
    }
    if d.get("common_mistake"):
        out["common_mistake"] = d["common_mistake"]
    if d.get("napkin_math"):
        out["napkin_math"] = d["napkin_math"]
    dd = d.get("deep_dive")
    if isinstance(dd, dict) and dd.get("url", "").startswith("https://"):
        out["deep_dive"] = {
            "title": dd.get("title", ""),
            "url": dd["url"],
        }
    return out


def _build_question(record: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
    """Return (question_dict, track, level, zone) for one record."""
    track = _coerce_track(record.get("track"))
    level = _coerce_level(record.get("level"))
    zone = _coerce_zone(record.get("zone"))

    q: dict[str, Any] = {
        "schema_version": 1,
        "id": record["id"],
        "title": record.get("title") or record["id"],
        "topic": record.get("topic") or "uncategorized",
        "status": _coerce_status(record.get("status")),
        "provenance": _infer_provenance(record),
        "scenario": record.get("scenario") or "",
        "details": _extract_details(record),
    }

    chain = _extract_chain(record)
    if chain:
        q["chain"] = chain

    # Preserve created_at/last_modified if the corpus has them (most records don't).
    if record.get("created_at"):
        q["created_at"] = record["created_at"]
    if record.get("validation_date"):
        q["last_modified"] = record["validation_date"]

    tags = []
    for key in ("hardware", "model_size", "scope"):
        v = record.get(key)
        if v:
            tags.append(f"{key}:{v}")
    if tags:
        q["tags"] = tags

    return q, track, level, zone


def _filename_for(qid: str) -> str:
    """Phase-1 filename for legacy IDs: preserve the ID as the stem.

    New IDs minted by ``vault new`` will use the content-addressed format
    ``<topic>-<6-hex>-<seq>.yaml``. Legacy IDs keep their original shape to
    preserve student bookmarks.
    """
    return f"{qid.lower()}.yaml"


def write_question(
    q: dict[str, Any], track: str, level: str, zone: str, *, force: bool = False
) -> tuple[Path, bool]:
    out_dir = QUESTIONS_ROOT / track / level / zone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _filename_for(q["id"])
    if out_path.exists() and not force:
        return out_path, False
    out_path.write_text(dump_str(q), encoding="utf-8")
    return out_path, True


def append_registry_entries(entries: list[dict[str, str]]) -> None:
    """Append (id, created_at, created_by) entries to id-registry.yaml."""
    existing: list[str] = []
    if REGISTRY_PATH.exists():
        existing = REGISTRY_PATH.read_text(encoding="utf-8").splitlines()

    # First-write preamble.
    if not existing:
        existing = [
            "# id-registry.yaml — APPEND-ONLY log of every ID ever assigned.",
            "# Never rewrite this file. CI rejects line deletions.",
            "# Format: one YAML mapping per line.",
            "entries:",
        ]
    known_ids = set()
    for line in existing:
        m = re.search(r"\bid:\s*([A-Za-z0-9._-]+)", line)
        if m:
            known_ids.add(m.group(1))

    for e in entries:
        if e["id"] in known_ids:
            continue
        existing.append(
            f"  - {{id: {e['id']}, created_at: {e['created_at']}, created_by: {e['created_by']}}}"
        )
        known_ids.add(e["id"])

    REGISTRY_PATH.write_text("\n".join(existing) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Split corpus.json into per-question YAML files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing YAML files.")
    parser.add_argument("--limit", type=int, default=None, help="Split only N records (for smoke tests).")
    args = parser.parse_args(argv)

    if not CORPUS_JSON.exists():
        sys.stderr.write(f"error: {CORPUS_JSON} not found\n")
        return 3

    corpus = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    if args.limit:
        corpus = corpus[: args.limit]

    now = datetime.now(UTC).isoformat(timespec="seconds")
    created_by = "split-corpus.py"

    written = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    registry_entries: list[dict[str, str]] = []

    for rec in corpus:
        try:
            q, track, level, zone = _build_question(rec)
            path, did_write = write_question(q, track, level, zone, force=args.force)
            if did_write:
                written += 1
            else:
                skipped += 1
            registry_entries.append({"id": q["id"], "created_at": now, "created_by": created_by})
        except Exception as exc:  # noqa: BLE001
            errors.append((rec.get("id", "<unknown>"), str(exc)))

    append_registry_entries(registry_entries)

    print(f"split complete: {written} written, {skipped} skipped, {len(errors)} errors")
    if errors:
        for qid, msg in errors[:20]:
            print(f"  error: {qid}: {msg}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
