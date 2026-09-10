#!/usr/bin/env python3
"""Topic resolver — maps old taxonomy fields to the new 79-topic system.

This is the bridge between the old corpus (primary_concept, reasoning_mode,
reasoning_competency, knowledge_area) and the new system (topic, zone).

Usage:
    # As a library
    from schema.resolve import resolve_topic, resolve_zone, migrate_question

    # As a CLI — dry-run migration on the whole corpus
    python3 resolve.py                    # Show mapping stats
    python3 resolve.py --apply            # Write topic + zone into corpus.json
    python3 resolve.py --question cloud-0042  # Show mapping for one question
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

VAULT = Path(__file__).resolve().parent.parent
CORPUS_PATH = VAULT / "corpus.json"
TAXONOMY_DATA = Path(__file__).resolve().parent / "taxonomy_data.yaml"

from zones import (
    REASONING_MODE_TO_ZONE,
    ZONE_LEVEL_AFFINITY,
    ALL_ZONES,
)

# ── Load taxonomy ────────────────────────────────────────────

def load_topics() -> dict[str, dict]:
    """Load the 79 topics as {id: topic_dict}."""
    with open(TAXONOMY_DATA) as f:
        data = yaml.safe_load(f)
    return {t["id"]: t for t in data["topics"]}


def build_concept_to_topic_map(topics: dict[str, dict]) -> dict[str, str]:
    """Build a mapping from old primary_concept values to new topic IDs.

    Strategy:
    1. Exact match (concept ID == topic ID)
    2. Prefix match (concept starts with a topic ID)
    3. Area + keyword match (fallback)
    """
    mapping = {}

    # All topic IDs are exact matches
    for tid in topics:
        mapping[tid] = tid

    # Build keyword index: for each topic, extract key words from name
    topic_keywords = {}
    for tid, t in topics.items():
        words = set(t["name"].lower().replace("&", "").replace("/", " ").split())
        words.discard("the")
        words.discard("and")
        words.discard("for")
        words.discard("a")
        topic_keywords[tid] = words

    return mapping, topic_keywords


def resolve_topic(concept: str, area: str, topics: dict,
                  mapping: dict, keywords: dict) -> str | None:
    """Resolve a primary_concept + competency_area to a topic ID."""
    if not concept:
        return None

    # 1. Exact match
    if concept in mapping:
        return mapping[concept]

    # 2. Prefix match — concept starts with a topic ID
    for tid in sorted(topics.keys(), key=len, reverse=True):
        if concept.startswith(tid):
            return tid

    # 3. Substring match — topic ID is contained in concept
    for tid in sorted(topics.keys(), key=len, reverse=True):
        if tid in concept:
            return tid

    # 4. Area match — find best topic in same area by keyword overlap
    concept_words = set(concept.lower().replace("-", " ").split())
    best_tid = None
    best_score = 0
    for tid, t in topics.items():
        if t["area"] != area and area:
            continue
        score = len(concept_words & keywords[tid])
        if score > best_score:
            best_score = score
            best_tid = tid

    if best_score >= 1:
        return best_tid

    # 5. Last resort — pick the most general topic in the area
    area_topics = [tid for tid, t in topics.items() if t["area"] == area]
    if area_topics:
        return area_topics[0]

    return None


def resolve_zone(reasoning_mode: str | None, level: str | None) -> str:
    """Resolve old reasoning_mode + level to a new zone."""
    # 1. Direct mapping from reasoning_mode
    if reasoning_mode and reasoning_mode in REASONING_MODE_TO_ZONE:
        return REASONING_MODE_TO_ZONE[reasoning_mode]

    # 2. Infer from level
    if level:
        if level in ("L1", "L2"):
            return "recall"
        elif level == "L3":
            return "fluency"
        elif level == "L4":
            return "diagnosis"
        elif level == "L5":
            return "evaluation"
        elif level in ("L6", "L6+"):
            return "mastery"

    return "recall"


def migrate_question(q: dict, topics: dict, mapping: dict,
                     keywords: dict) -> dict:
    """Add topic and zone fields to a question dict."""
    concept = q.get("primary_concept", "") or q.get("taxonomy_concept", "")
    area = q.get("competency_area", "")
    mode = q.get("reasoning_mode")
    level = q.get("level")

    # Normalize area
    area_norm = area.lower().replace(" ", "-") if area else ""
    if area_norm not in {t["area"] for t in topics.values()}:
        # Try common normalizations
        AREA_MAP = {
            "mlops": "deployment", "serving-systems": "deployment",
            "distributed-training": "parallelism", "distributed": "parallelism",
            "security": "reliability", "safety": "reliability",
            "sustainability": "power", "inference": "latency",
            "performance": "latency", "benchmarking": "latency",
            "frameworks": "optimization", "compilation": "optimization",
            "model-architecture": "architecture", "data-engineering": "data",
            "economics": "cross-cutting", "monitoring": "reliability",
            "fault-tolerance": "reliability",
        }
        area_norm = AREA_MAP.get(area_norm, area_norm)

    topic = resolve_topic(concept, area_norm, topics, mapping, keywords)
    zone = resolve_zone(mode, level)

    result = dict(q)
    if topic:
        result["topic"] = topic
    result["zone"] = zone
    return result


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Migrate corpus to new topic+zone system")
    parser.add_argument("--apply", action="store_true",
                        help="Write topic and zone fields into corpus.json")
    parser.add_argument("--question", help="Show mapping for a single question ID")
    args = parser.parse_args()

    topics = load_topics()
    mapping, keywords = build_concept_to_topic_map(topics)

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Corpus: {len(corpus)} questions")
    print(f"Topics: {len(topics)}")
    print()

    if args.question:
        q = next((q for q in corpus if q["id"] == args.question), None)
        if not q:
            print(f"Question '{args.question}' not found")
            sys.exit(1)

        migrated = migrate_question(q, topics, mapping, keywords)
        print(f"Question: {q['id']}")
        print(f"  Title: {q.get('title', '?')}")
        print(f"  Track: {q.get('track', '?')}")
        print(f"  Level: {q.get('level', '?')}")
        print(f"  Old primary_concept: {q.get('primary_concept', '?')}")
        print(f"  Old competency_area: {q.get('competency_area', '?')}")
        print(f"  Old reasoning_mode:  {q.get('reasoning_mode', '?')}")
        print(f"  → topic: {migrated.get('topic', 'UNMAPPED')}")
        print(f"  → zone:  {migrated.get('zone', 'UNMAPPED')}")
        return

    # Migrate all questions and collect stats
    mapped = 0
    unmapped = 0
    topic_counts = Counter()
    zone_counts = Counter()
    unmapped_concepts = Counter()

    for q in corpus:
        migrated = migrate_question(q, topics, mapping, keywords)
        if migrated.get("topic"):
            mapped += 1
            topic_counts[migrated["topic"]] += 1
        else:
            unmapped += 1
            unmapped_concepts[q.get("primary_concept", "EMPTY")] += 1
        zone_counts[migrated["zone"]] += 1

    print(f"Mapped:   {mapped} ({100*mapped/len(corpus):.1f}%)")
    print(f"Unmapped: {unmapped} ({100*unmapped/len(corpus):.1f}%)")

    print(f"\nTopic distribution (top 20):")
    for t, cnt in topic_counts.most_common(20):
        name = topics[t]["name"]
        print(f"  {name:40s} {cnt:>5}")

    print(f"\nZone distribution:")
    for z, cnt in sorted(zone_counts.items(), key=lambda x: -x[1]):
        print(f"  {z:15s} {cnt:>5}")

    if unmapped_concepts:
        print(f"\nTop unmapped concepts:")
        for c, cnt in unmapped_concepts.most_common(15):
            print(f"  {c}: {cnt}")

    # Coverage matrix: topic × zone
    print(f"\nTopic × Zone coverage (topics with questions in 3+ zones):")
    topic_zones = defaultdict(set)
    for q in corpus:
        m = migrate_question(q, topics, mapping, keywords)
        if m.get("topic"):
            topic_zones[m["topic"]].add(m["zone"])
    multi_zone = {t: zones for t, zones in topic_zones.items() if len(zones) >= 3}
    for t in sorted(multi_zone, key=lambda x: -len(multi_zone[x]))[:15]:
        name = topics[t]["name"]
        zones = ", ".join(sorted(multi_zone[t]))
        print(f"  {name:40s} [{len(multi_zone[t])} zones] {zones}")

    if args.apply:
        print(f"\nApplying migration to corpus.json...")
        for i, q in enumerate(corpus):
            migrated = migrate_question(q, topics, mapping, keywords)
            if migrated.get("topic"):
                corpus[i]["topic"] = migrated["topic"]
            corpus[i]["zone"] = migrated["zone"]

        with open(CORPUS_PATH, "w") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  Done. {mapped} questions now have topic + zone fields.")


if __name__ == "__main__":
    main()
