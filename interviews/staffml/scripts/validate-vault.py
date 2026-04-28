#!/usr/bin/env python3
"""Validate vault data integrity for StaffML deployment.

When ``corpus.json`` is present (e.g. after ``vault build --legacy-json``), runs
full cross-checks against taxonomy and manifest.

When ``corpus.json`` is absent — the normal case for a clean clone after
2026-04-26, when corpus was retired as a tracked file — runs **sparse** checks
only: committed ``taxonomy.json`` and ``vault-manifest.json`` must load and
look self-consistent. Full per-question validation is expected from
``vault check --strict`` in CI (``staffml-validate-vault.yml``) and from this
script after a local or CI ``vault build -- ... --legacy-json``.

Exit code 0 = all checks pass, 1 = errors found.

Usage: python3 interviews/staffml/scripts/validate-vault.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

STAFFML_DATA = Path(__file__).parent.parent / "src" / "data"

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


def run_sparse_validation(taxonomy_path: Path, manifest_path: Path) -> int:
    """Validate committed JSON when the full bundled corpus is not on disk."""
    print("\n🔍 Sparse mode (no corpus.json)")
    print(
        "   Per-question checks require a build artifact. Regenerate with:\n"
        "   vault build --vault-dir interviews/vault --release-id <id> --legacy-json\n"
    )

    if not taxonomy_path.exists():
        error(f"taxonomy.json not found at {taxonomy_path}")
        return 1
    if not manifest_path.exists():
        error(f"vault-manifest.json not found at {manifest_path}")
        return 1

    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = json.load(f)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    concepts = taxonomy.get("concepts", [])
    if not concepts:
        error("taxonomy has no concepts")
    else:
        ok(f"taxonomy: {len(concepts)} concepts")

    qc = manifest.get("questionCount")
    if qc is None or not isinstance(qc, int) or qc < 1:
        error("manifest.questionCount missing or invalid")
    else:
        ok(f"manifest: questionCount = {qc}")

    td = manifest.get("trackDistribution") or {}
    if isinstance(td, dict) and td:
        s = sum(int(v) for v in td.values() if isinstance(v, int))
        if s != qc:
            warn(
                f"trackDistribution sum ({s}) != questionCount ({qc}) — check manifest"
            )
        else:
            ok("trackDistribution sums to questionCount")

    # Release identity is the single source of truth for the displayed
    # version on the site. A missing or malformed value is a hard error
    # — the bundle MUST not ship without a labeled release.
    rid = manifest.get("releaseId")
    rhash = manifest.get("releaseHash")
    if not rid or not isinstance(rid, str):
        error("manifest.releaseId missing or invalid")
    if not rhash or not isinstance(rhash, str) or len(rhash) < 16:
        error("manifest.releaseHash missing or not a full hex digest")
    if rid and rhash:
        ok(f"Vault v{rid} — hash {rhash[:7]}")

    print(f"\n{'=' * 50}")
    print(f"  Mode:     sparse (no corpus.json)")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"{'=' * 50}")

    if errors:
        print("\n❌ Sparse validation failed")
        return 1
    print(
        "\n🎯 Sparse checks passed — for full deploy-grade validation, run vault build "
        "--legacy-json and re-run this script, or rely on staffml-validate-vault (CI)."
    )
    if warnings:
        print(f"   ({len(warnings)} warnings — review recommended)")
    return 0


# ── 1. Load data ─────────────────────────────────────────────

corpus_path = STAFFML_DATA / "corpus.json"
taxonomy_path = STAFFML_DATA / "taxonomy.json"
manifest_path = STAFFML_DATA / "vault-manifest.json"

if not corpus_path.exists():
    sys.exit(run_sparse_validation(taxonomy_path, manifest_path))

if not taxonomy_path.exists():
    print(f"  ❌ taxonomy.json not found at {taxonomy_path}", file=sys.stderr)
    sys.exit(1)

print("\n🔍 Loading data files...")

with open(corpus_path, encoding="utf-8") as f:
    corpus = json.load(f)
with open(taxonomy_path, encoding="utf-8") as f:
    taxonomy = json.load(f)

manifest = None
if manifest_path.exists():
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

ok(f"Loaded {len(corpus)} questions, {len(taxonomy.get('concepts', []))} concepts")

# ── 2. Schema checks ─────────────────────────────────────────

print("\n📋 Schema validation...")

REQUIRED_FIELDS = [
    "id",
    "title",
    "level",
    "track",
    "scenario",
    "competency_area",
    "details",
]
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

    for field in REQUIRED_FIELDS:
        if not q.get(field):
            error(f"{qid}: missing required field '{field}'")
            missing_fields += 1

    if q.get("level") not in VALID_LEVELS:
        error(f"{qid}: invalid level '{q.get('level')}'")
        bad_levels += 1

    if q.get("track") not in VALID_TRACKS:
        error(f"{qid}: invalid track '{q.get('track')}'")
        bad_tracks += 1

    scenario = q.get("scenario", "")
    if len(scenario.strip()) < 30:
        warn(f"{qid}: scenario too short ({len(scenario)} chars)")
        short_scenarios += 1

    details = q.get("details", {})
    for df in DETAIL_FIELDS:
        if not details.get(df) or len(str(details.get(df, "")).strip()) < 5:
            warn(f"{qid}: details.{df} empty or too short")
            empty_answers += 1

if missing_fields == 0 and bad_levels == 0 and bad_tracks == 0:
    ok("All questions have valid required fields, levels, and tracks")
else:
    error(
        f"{missing_fields} missing fields, {bad_levels} bad levels, {bad_tracks} bad tracks"
    )

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

duplicate_chains = 0
for cid, qs in chains.items():
    pos_list = []
    for q in qs:
        cp = q.get("chain_positions", -1)
        if isinstance(cp, dict):
            pos_list.append(int(cp.get(cid, -1)))
        else:
            pos_list.append(int(cp) if cp != "" else -1)
    if len(pos_list) != len(set(pos_list)):
        duplicate_chains += 1
        if duplicate_chains <= 3:
            warn(f"Chain '{cid}': duplicate positions {sorted(pos_list)}")

if duplicate_chains == 0:
    ok(f"All {len(chains)} chains have unique positions")
else:
    warn(f"{duplicate_chains} chains have duplicate positions")

# ── 6. Manifest consistency ──────────────────────────────────

print("\n📦 Manifest consistency...")

if manifest:
    if manifest.get("questionCount") != len(corpus):
        error(
            f"Manifest says {manifest['questionCount']} questions, corpus has {len(corpus)}"
        )
    else:
        ok(f"Manifest matches corpus: {len(corpus)} questions")

    if manifest.get("chainCount") != len(chains):
        warn(
            f"Manifest says {manifest['chainCount']} chains, found {len(chains)}"
        )

    rid = manifest.get("releaseId")
    rhash = manifest.get("releaseHash")
    if not rid or not isinstance(rid, str):
        error("manifest.releaseId missing or invalid")
    elif not rhash or not isinstance(rhash, str) or len(rhash) < 16:
        error("manifest.releaseHash missing or not a full hex digest")
    else:
        ok(f"Vault v{rid} — hash {rhash[:7]}")
else:
    warn("No vault-manifest.json found — run vault build --legacy-json")

# ── 7. Distribution sanity ───────────────────────────────────

print("\n📊 Distribution sanity...")

level_dist = Counter(q.get("level") for q in corpus)
track_dist = Counter(q.get("track") for q in corpus)

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
print("\n🎯 All checks passed — vault is deployment-ready")
if warnings:
    print(f"   ({len(warnings)} warnings — review recommended)")
sys.exit(0)
