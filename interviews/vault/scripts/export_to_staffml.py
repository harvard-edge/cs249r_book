#!/usr/bin/env python3
# ┌─── DEPRECATED ────────────────────────────────────────────────────────────┐
# │ Pre-YAML-migration script. Replaced by:                                   │
# │     vault build --local-json                                              │
# │ See ./DEPRECATED.md for the full map.                                     │
# └───────────────────────────────────────────────────────────────────────────┘
"""Export vault corpus to StaffML web app format.

Converts vault data files to the JSON format consumed by the Next.js app:
1. corpus.json    — published questions with validation fields stripped
2. taxonomy.json  — topics grouped by area (replaces old 825-concept taxonomy)
3. topics.json    — curated topic graph with prerequisites
4. zones.json     — ikigai zone definitions
5. corpus-index.json — aggregate counts for fast UI rendering

Usage:
    python3 vault/scripts/export_to_staffml.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VAULT_DIR = Path(__file__).resolve().parent.parent
STAFFML_DATA = VAULT_DIR.parent / "staffml" / "src" / "data"

# Fields to keep in the exported corpus (strip validation internals)
KEEP_FIELDS = {
    "id", "title", "track", "level", "scope",
    "topic", "zone", "competency_area", "bloom_level",
    "scenario", "details",
    "chain_ids", "chain_positions",
}


def export_corpus() -> tuple[list[dict], dict]:
    """Load vault corpus, filter to published, strip internal fields."""
    with open(VAULT_DIR / "corpus.json") as f:
        raw = json.load(f)

    published = []
    topic_counts: dict[str, int] = {}
    zone_counts: dict[str, int] = {}
    area_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    track_counts: dict[str, int] = {}

    for q in raw:
        if not q.get("validated"):
            continue

        # Build clean question
        clean: dict = {}
        for k in KEEP_FIELDS:
            if k in q:
                clean[k] = q[k]
        published.append(clean)

        # Accumulate stats
        t = q.get("topic", "unknown")
        topic_counts[t] = topic_counts.get(t, 0) + 1
        z = q.get("zone", "unknown")
        zone_counts[z] = zone_counts.get(z, 0) + 1
        a = q.get("competency_area", "unknown")
        area_counts[a] = area_counts.get(a, 0) + 1
        lv = q.get("level", "unknown")
        level_counts[lv] = level_counts.get(lv, 0) + 1
        tr = q.get("track", "unknown")
        track_counts[tr] = track_counts.get(tr, 0) + 1

    index = {
        "total": len(published),
        "topics": dict(sorted(topic_counts.items())),
        "zones": dict(sorted(zone_counts.items())),
        "areas": dict(sorted(area_counts.items())),
        "levels": dict(sorted(level_counts.items())),
        "tracks": dict(sorted(track_counts.items())),
    }

    return published, index


def export_taxonomy() -> dict:
    """Build taxonomy.json from topics.json — topics grouped by area."""
    with open(VAULT_DIR / "topics.json") as f:
        topics_data = json.load(f)

    # Load corpus for per-topic question counts and level distribution
    with open(VAULT_DIR / "corpus.json") as f:
        corpus = json.load(f)

    published = [q for q in corpus if q.get("validated")]

    # Build per-topic stats
    topic_stats: dict[str, dict] = {}
    for q in published:
        tid = q.get("topic", "")
        if tid not in topic_stats:
            topic_stats[tid] = {"question_count": 0, "level_distribution": {}, "tracks": set()}
        topic_stats[tid]["question_count"] += 1
        lv = q.get("level", "")
        topic_stats[tid]["level_distribution"][lv] = topic_stats[tid]["level_distribution"].get(lv, 0) + 1
        topic_stats[tid]["tracks"].add(q.get("track", ""))

    # Build concepts list enriched with stats
    concepts = []
    for topic in topics_data["topics"]:
        tid = topic["id"]
        stats = topic_stats.get(tid, {"question_count": 0, "level_distribution": {}, "tracks": set()})
        concepts.append({
            "id": tid,
            "name": topic["name"],
            "description": topic.get("description", ""),
            "area": topic["area"],
            "prerequisites": topic.get("prerequisites", []),
            "tracks": sorted(stats["tracks"]) if isinstance(stats["tracks"], set) else stats["tracks"],
            "question_count": stats["question_count"],
            "level_distribution": stats["level_distribution"],
        })

    return {
        "version": topics_data.get("version", "1.0.0"),
        "areas": topics_data["areas"],
        "concepts": concepts,
    }


def export_zones() -> dict:
    """Convert zones.py definitions to JSON."""
    # Import directly from the zones module
    sys.path.insert(0, str(VAULT_DIR / "schema"))
    from zones import ZONE_SKILLS, ZONE_DESCRIPTIONS, ZONE_LEVEL_AFFINITY

    zones = {}
    for zone_id, skills in ZONE_SKILLS.items():
        zones[zone_id] = {
            "skills": sorted(skills),
            "description": ZONE_DESCRIPTIONS.get(zone_id, ""),
            "levels": ZONE_LEVEL_AFFINITY.get(zone_id, []),
        }

    return {
        "version": "1.0.0",
        "zones": zones,
    }


def main():
    print("Exporting vault → StaffML app data...")

    # 1. Corpus
    published, index = export_corpus()
    corpus_path = STAFFML_DATA / "corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(published, f, indent=2)
    print(f"  ✓ corpus.json: {len(published)} published questions")

    # 2. Taxonomy
    taxonomy = export_taxonomy()
    taxonomy_path = STAFFML_DATA / "taxonomy.json"
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"  ✓ taxonomy.json: {len(taxonomy['concepts'])} topics across {len(taxonomy['areas'])} areas")

    # 3. Topics (copy directly — already clean JSON)
    import shutil
    topics_src = VAULT_DIR / "topics.json"
    topics_dst = STAFFML_DATA / "topics.json"
    shutil.copy2(topics_src, topics_dst)
    print(f"  ✓ topics.json: copied from vault")

    # 4. Zones
    zones = export_zones()
    zones_path = STAFFML_DATA / "zones.json"
    with open(zones_path, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"  ✓ zones.json: {len(zones['zones'])} zones")

    # 5. Corpus index
    index_path = STAFFML_DATA / "corpus-index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  ✓ corpus-index.json: aggregate counts")

    print(f"\nDone. All files written to {STAFFML_DATA}")


if __name__ == "__main__":
    main()
