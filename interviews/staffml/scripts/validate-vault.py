#!/usr/bin/env python3
"""Validate vault data integrity for StaffML deployment.

Checks that corpus.json, taxonomy.json, and vault-manifest.json are
consistent with each other and with the app's expectations.

Exit code 0 = all checks pass, 1 = errors found.

Usage: python3 interviews/staffml/scripts/validate-vault.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

STAFFML_DATA = Path(__file__).parent.parent / "src" / "data"
VAULT_DIR = Path(__file__).parent.parent.parent / "vault"

errors: list[str] = []
warnings: list[str] = []


def error(msg: str) -> None:
    errors.append(msg)
    print(f"  ❌ {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  ⚠️  {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


# ── 1. Load data ─────────────────────────────────────────────

print("\n🔍 Loading data files...")

corpus_path = STAFFML_DATA / "corpus.json"
taxonomy_path = STAFFML_DATA / "taxonomy.json"
manifest_path = STAFFML_DATA / "vault-manifest.json"

if not corpus_path.exists():
    error(f"corpus.json not found at {corpus_path}")
    sys.exit(1)
if not taxonomy_path.exists():
    error(f"taxonomy.json not found at {taxonomy_path}")
    sys.exit(1)

with open(corpus_path) as f:
    corpus = json.load(f)
with open(taxonomy_path) as f:
    taxonomy = json.load(f)

manifest = None
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)

ok(f"Loaded {len(corpus)} questions, {len(taxonomy.get('concepts', []))} concepts")

# ── 2. Schema checks ─────────────────────────────────────────

print("\n📋 Schema validation...")

REQUIRED_FIELDS = ["id", "title", "level", "track", "scenario", "competency_area", "details"]
VALID_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6", "L6+"}
VALID_TRACKS = {"cloud", "edge", "mobile", "tinyml", "global"}
DETAIL_FIELDS = ["common_mistake", "realistic_solution"]

missing_fields = 0
bad_levels = 0
bad_tracks = 0
short_scenarios = 0
empty_answers = 0

for q in corpus:
    qid = q.get("id", "???")

    # Required fields
    for field in REQUIRED_FIELDS:
        if not q.get(field):
            error(f"{qid}: missing required field '{field}'")
            missing_fields += 1

    # Valid level
    if q.get("level") not in VALID_LEVELS:
        error(f"{qid}: invalid level '{q.get('level')}'")
        bad_levels += 1

    # Valid track
    if q.get("track") not in VALID_TRACKS:
        error(f"{qid}: invalid track '{q.get('track')}'")
        bad_tracks += 1

    # Scenario quality
    scenario = q.get("scenario", "")
    if len(scenario.strip()) < 30:
        warn(f"{qid}: scenario too short ({len(scenario)} chars)")
        short_scenarios += 1

    # Answer quality
    details = q.get("details", {})
    for df in DETAIL_FIELDS:
        if not details.get(df) or len(str(details.get(df, "")).strip()) < 5:
            warn(f"{qid}: details.{df} empty or too short")
            empty_answers += 1

if missing_fields == 0 and bad_levels == 0 and bad_tracks == 0:
    ok("All questions have valid required fields, levels, and tracks")
else:
    error(f"{missing_fields} missing fields, {bad_levels} bad levels, {bad_tracks} bad tracks")

# ── 3. Uniqueness checks ─────────────────────────────────────

print("\n🔑 Uniqueness checks...")

ids = [q["id"] for q in corpus]
id_counts = Counter(ids)
dupes = {k: v for k, v in id_counts.items() if v > 1}
if dupes:
    error(f"{len(dupes)} duplicate IDs: {list(dupes.keys())[:5]}")
else:
    ok(f"All {len(ids)} question IDs are unique")

# ── 4. Taxonomy consistency ──────────────────────────────────

print("\n🏷️  Taxonomy consistency...")

concepts = {c["id"] for c in taxonomy.get("concepts", [])}
corpus_concepts = {q.get("taxonomy_concept") for q in corpus if q.get("taxonomy_concept")}
unmapped = corpus_concepts - concepts

if unmapped:
    warn(f"{len(unmapped)} corpus concepts not in taxonomy: {list(unmapped)[:5]}")
else:
    ok(f"All {len(corpus_concepts)} corpus concepts exist in taxonomy")

# Check competency areas used in corpus
corpus_areas = Counter(q.get("competency_area", "???") for q in corpus)
ok(f"{len(corpus_areas)} competency areas in use")

# ── 5. Chain integrity ───────────────────────────────────────

print("\n🔗 Chain integrity...")

chains: dict[str, list] = {}
for q in corpus:
    cids = q.get("chain_ids", "")
    if isinstance(cids, list):
        for cid in cids:
            if cid:
                chains.setdefault(cid, []).append(q)
    elif cids:
        chains.setdefault(cids, []).append(q)

solo_chains = sum(1 for c in chains.values() if len(c) <= 1)
if solo_chains > 0:
    warn(f"{solo_chains} single-question chains (should be 2+)")

# Check chain positions are sequential
bad_chains = 0
for cid, qs in chains.items():
    pos_list = []
    for q in qs:
        cp = q.get("chain_positions", -1)
        if isinstance(cp, dict):
            pos_list.append(int(cp.get(cid, -1)))
        else:
            pos_list.append(int(cp) if cp != "" else -1)
    positions = sorted(pos_list)
    expected = list(range(len(qs)))
    if positions != expected:
        bad_chains += 1
        if bad_chains <= 3:
            warn(f"Chain '{cid}': positions {positions} != expected {expected}")

if bad_chains == 0:
    ok(f"All {len(chains)} chains have sequential positions")
else:
    warn(f"{bad_chains} chains have non-sequential positions")

# ── 6. Manifest consistency ──────────────────────────────────

print("\n📦 Manifest consistency...")

if manifest:
    if manifest.get("questionCount") != len(corpus):
        error(f"Manifest says {manifest['questionCount']} questions, corpus has {len(corpus)}")
    else:
        ok(f"Manifest matches corpus: {len(corpus)} questions")

    if manifest.get("chainCount") != len(chains):
        warn(f"Manifest says {manifest['chainCount']} chains, found {len(chains)}")

    ok(f"Vault v{manifest.get('version', '?')} — hash {manifest.get('contentHash', '?')}")
else:
    warn("No vault-manifest.json found — run vault build --legacy-json")

# ── 7. Distribution sanity ───────────────────────────────────

print("\n📊 Distribution sanity...")

level_dist = Counter(q.get("level") for q in corpus)
track_dist = Counter(q.get("track") for q in corpus)

# Check no track has < 5% of total
for track, count in track_dist.items():
    pct = count / len(corpus) * 100
    if pct < 2:
        warn(f"Track '{track}' has only {count} questions ({pct:.1f}%)")

ok(f"Levels: {dict(sorted(level_dist.items()))}")
ok(f"Tracks: {dict(sorted(track_dist.items()))}")

# ── Summary ──────────────────────────────────────────────────

print(f"\n{'=' * 50}")
print(f"  Questions: {len(corpus)}")
print(f"  Chains:    {len(chains)}")
print(f"  Concepts:  {len(concepts)}")
print(f"  Errors:    {len(errors)}")
print(f"  Warnings:  {len(warnings)}")
print(f"{'=' * 50}")

if errors:
    print(f"\n❌ {len(errors)} errors found — vault is NOT deployment-ready")
    sys.exit(1)
else:
    print(f"\n🎯 All checks passed — vault is deployment-ready")
    if warnings:
        print(f"   ({len(warnings)} warnings — review recommended)")
    sys.exit(0)
