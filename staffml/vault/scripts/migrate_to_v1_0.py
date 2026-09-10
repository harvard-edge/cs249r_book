"""One-time migration: corpus.json -> per-question YAML at schema_version 1.0.

This script replaces the `split_corpus.py` migration that introduced the
bugs fixed here. It writes one YAML file per question into a flat-by-track
layout:

    questions/<track>/<id>.yaml

Every axis (track, level, zone, topic, competency_area, bloom_level) lives
inside the YAML. The filesystem carries only `track` for navigability; a
fast-tier invariant validates that the filename prefix matches `yaml.track`.

Usage:
    python3 scripts/migrate_to_v1_0.py --dry-run                    # preview stats
    python3 scripts/migrate_to_v1_0.py --staging /tmp/vault-stage   # write to staging
    python3 scripts/migrate_to_v1_0.py --apply                      # write to questions/

After --apply, the old {track}/{level}/{zone}/ hierarchy is removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0"

SCRIPT_DIR = Path(__file__).resolve().parent
VAULT_DIR = SCRIPT_DIR.parent
CORPUS_PATH = VAULT_DIR / "corpus.json"
QUESTIONS_DIR = VAULT_DIR / "questions"

VALID_TRACKS = {"cloud", "edge", "mobile", "tinyml", "global"}
VALID_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6+"}
VALID_ZONES = {
    "recall", "fluency", "analyze", "design", "implement",
    "diagnosis", "specification", "optimization",
    "evaluation", "realization", "mastery",
}

# Bloom's taxonomy was revised in 2001; "synthesize" was replaced by "create".
BLOOM_NORMALIZATION = {"synthesize": "create"}

# Fields dropped during v1.0 migration (unused by GUI, unpopulated, or dead).
DROPPED_FIELDS = {"scope", "mode", "version"}

# Canonical field order for schema_version 1.0 YAML output.
# Groups: identity, classification, content, workflow, chains, validation,
# math validation, human review, metadata.
FIELD_ORDER = [
    "schema_version",
    "id",
    # --- classification (4-axis + Bloom's + phase) ---
    "track",
    "level",
    "zone",
    "topic",
    "competency_area",
    "bloom_level",
    "phase",
    # --- content ---
    "title",
    "scenario",
    "details",
    # --- workflow ---
    "status",
    "provenance",
    "requires_explanation",
    "expected_time_minutes",
    "deletion_reason",
    # --- chain membership ---
    "chains",
    # --- LLM validation lineage (Gemini review) ---
    "validated",
    "validation_status",
    "validation_date",
    "validation_model",
    "validation_issues",
    "validation_status_pro",
    "validation_issues_pro",
    # --- math validation ---
    "math_verified",
    "math_status",
    "math_date",
    "math_model",
    "math_issues",
    # --- human review (new in v1.0 — tracks human verification separately from LLM) ---
    "human_reviewed",
    # --- review notes (Pro-model classification feedback, 31 questions) ---
    "classification_review",
    # --- tags + temporal ---
    "tags",
    "created_at",
    "updated_at",
    "last_modified",
]

DETAILS_ORDER = ["scenario", "realistic_solution", "common_mistake", "napkin_math",
                 "options", "correct_index", "resources"]


class OrderedDumper(yaml.SafeDumper):
    """Preserve dict insertion order and use block style for readability."""


def _represent_dict_preserve_order(dumper: yaml.SafeDumper, data: dict) -> yaml.Node:
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        data.items(),
        flow_style=False,
    )


OrderedDumper.add_representer(dict, _represent_dict_preserve_order)


def is_empty(value: Any) -> bool:
    """Omit fields with empty-ish values for clean YAML."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def transform_chains(q: dict) -> list[dict] | None:
    """corpus.json has chain_ids + chain_positions (parallel structures).
    YAML uses chains: [{id, position}, ...] — plural, ordered by position.
    """
    chain_ids = q.get("chain_ids") or []
    positions = q.get("chain_positions") or {}
    if not chain_ids:
        return None
    chains = []
    for cid in chain_ids:
        entry = {"id": cid}
        if cid in positions:
            entry["position"] = positions[cid]
        chains.append(entry)
    # Sort by (id, position) for stable output
    chains.sort(key=lambda e: (e["id"], e.get("position", 0)))
    return chains


def normalize_details(details: dict | None) -> dict:
    """Reorder details in canonical order, omit empty fields."""
    if not details:
        return {}
    out = {}
    for key in DETAILS_ORDER:
        if key in details:
            val = details[key]
            if not is_empty(val):
                out[key] = val
    # Preserve any unknown keys at the end (forward compat)
    for key, val in details.items():
        if key not in DETAILS_ORDER and not is_empty(val):
            out[key] = val
    return out


def normalize_bloom(bloom: str | None) -> str | None:
    """Bloom's taxonomy revised 2001: 'synthesize' -> 'create'."""
    if bloom is None:
        return None
    return BLOOM_NORMALIZATION.get(bloom, bloom)


def human_reviewed_scaffold() -> dict:
    """Seed the human_reviewed block. All current questions are LLM-only;
    a human review workflow populates this going forward."""
    return {
        "status": "not-reviewed",   # not-reviewed | verified | flagged | needs-rework
        "by": None,
        "date": None,
        "notes": None,
    }


def build_yaml_record(q: dict) -> dict:
    """Build a canonical schema-v1.0 YAML dict from a corpus.json question record."""
    raw = {
        "schema_version": SCHEMA_VERSION,
        "id": q.get("id"),
        "track": q.get("track"),
        "level": q.get("level"),
        "zone": q.get("zone"),
        "topic": q.get("topic"),
        "competency_area": q.get("competency_area"),
        "bloom_level": normalize_bloom(q.get("bloom_level")),
        "phase": q.get("phase"),
        "title": q.get("title"),
        "scenario": q.get("scenario"),
        "details": normalize_details(q.get("details")),
        "status": q.get("status"),
        "provenance": q.get("provenance", "imported"),
        "requires_explanation": q.get("requires_explanation"),
        "expected_time_minutes": q.get("expected_time_minutes"),
        "deletion_reason": q.get("deletion_reason"),
        "chains": transform_chains(q),
        "validated": q.get("validated"),
        "validation_status": q.get("validation_status"),
        "validation_date": q.get("validation_date"),
        "validation_model": q.get("validation_model"),
        "validation_issues": q.get("validation_issues"),
        "validation_status_pro": q.get("validation_status_pro"),
        "validation_issues_pro": q.get("validation_issues_pro"),
        "math_verified": q.get("math_verified"),
        "math_status": q.get("math_status"),
        "math_date": q.get("math_date"),
        "math_model": q.get("math_model"),
        "math_issues": q.get("math_issues"),
        "human_reviewed": human_reviewed_scaffold(),
        "classification_review": q.get("classification_review"),
        "tags": q.get("tags"),
        "created_at": q.get("created_at"),
        "updated_at": q.get("updated_at"),
        "last_modified": q.get("last_modified"),
    }

    # Emit fields in canonical order, omitting empty values
    out = {}
    for key in FIELD_ORDER:
        if key in raw:
            val = raw[key]
            # details is always present (even if just {}); but if empty, skip
            if is_empty(val) and key != "schema_version":
                # Preserve `validated: False` as a real value
                if key == "validated" and raw[key] is False:
                    out[key] = False
                continue
            out[key] = val

    return out


def validate_record(record: dict, orphan_topics_ok: set[str]) -> list[str]:
    """Return a list of validation errors (empty if clean)."""
    errors = []
    if record.get("track") not in VALID_TRACKS:
        errors.append(f"invalid track: {record.get('track')!r}")
    if record.get("level") not in VALID_LEVELS:
        errors.append(f"invalid level: {record.get('level')!r}")
    if record.get("zone") not in VALID_ZONES:
        errors.append(f"invalid zone: {record.get('zone')!r}")
    if not record.get("topic"):
        errors.append("missing topic")
    if not record.get("id"):
        errors.append("missing id")
    if not record.get("title"):
        errors.append("missing title")
    scenario = record.get("scenario") or ""
    if len(scenario.strip()) < 30:
        errors.append(f"scenario too short ({len(scenario)} chars)")
    return errors


def target_path(record: dict, output_dir: Path) -> Path:
    return output_dir / record["track"] / f"{record['id']}.yaml"


def write_yaml(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(
            record,
            fh,
            Dumper=OrderedDumper,
            sort_keys=False,
            allow_unicode=True,
            width=100,
            default_flow_style=False,
        )


def run_migration(
    corpus_path: Path,
    output_dir: Path,
    dry_run: bool,
    sample: int,
) -> dict:
    with corpus_path.open() as fh:
        corpus = json.load(fh)

    stats = {
        "total": len(corpus),
        "written": 0,
        "errors": 0,
        "samples": [],
        "by_track": Counter(),
        "by_level": Counter(),
        "by_zone": Counter(),
        "orphan_topics": Counter(),
    }
    error_log = []
    orphan_topics_ok: set[str] = set()  # schema v1.0 accepts all topics from corpus

    for q in corpus:
        record = build_yaml_record(q)
        errors = validate_record(record, orphan_topics_ok)
        if errors:
            error_log.append((q.get("id", "?"), errors))
            stats["errors"] += 1
            continue

        stats["by_track"][record["track"]] += 1
        stats["by_level"][record["level"]] += 1
        stats["by_zone"][record["zone"]] += 1

        path = target_path(record, output_dir)

        if not dry_run:
            write_yaml(record, path)

        if len(stats["samples"]) < sample:
            stats["samples"].append((record["id"], str(path), record["level"], record["zone"]))

        stats["written"] += 1

    stats["error_log"] = error_log[:20]
    return stats


def fresh_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate vault questions to schema v1.0")
    ap.add_argument("--dry-run", action="store_true",
                    help="report stats without writing any files")
    ap.add_argument("--staging", type=Path,
                    help="write to a staging directory instead of questions/")
    ap.add_argument("--apply", action="store_true",
                    help="replace questions/ with migrated output (destructive)")
    ap.add_argument("--sample", type=int, default=5,
                    help="number of sample paths to print")
    args = ap.parse_args()

    if sum(1 for x in (args.dry_run, args.staging, args.apply) if x) != 1:
        print("ERROR: specify exactly one of --dry-run, --staging, --apply", file=sys.stderr)
        return 2

    if args.dry_run:
        output_dir = Path("/tmp/__vault_migrate_dryrun__")  # sink
    elif args.staging:
        output_dir = args.staging
        fresh_output_dir(output_dir)
    else:  # --apply
        output_dir = QUESTIONS_DIR.parent / ".questions_new"
        fresh_output_dir(output_dir)

    stats = run_migration(
        corpus_path=CORPUS_PATH,
        output_dir=output_dir,
        dry_run=args.dry_run,
        sample=args.sample,
    )

    print(f"Migration: schema_version={SCHEMA_VERSION}")
    print(f"  corpus records: {stats['total']}")
    print(f"  written:        {stats['written']}")
    print(f"  errors:         {stats['errors']}")
    print()
    print("By track:", dict(stats["by_track"].most_common()))
    print("By level:", dict(stats["by_level"].most_common()))
    print("By zone: ", dict(stats["by_zone"].most_common()))
    print()
    print("Sample outputs:")
    for s in stats["samples"]:
        print(f"  {s}")
    if stats["error_log"]:
        print()
        print(f"First {len(stats['error_log'])} errors:")
        for qid, errs in stats["error_log"]:
            print(f"  {qid}: {errs}")

    if args.apply and stats["errors"] == 0:
        print()
        print(f"Swapping {QUESTIONS_DIR} with {output_dir} ...")
        backup = QUESTIONS_DIR.parent / ".questions_old"
        if backup.exists():
            shutil.rmtree(backup)
        QUESTIONS_DIR.rename(backup)
        output_dir.rename(QUESTIONS_DIR)
        print(f"Old tree preserved at: {backup}")
        print(f"New tree is at:        {QUESTIONS_DIR}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
