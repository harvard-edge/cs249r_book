#!/usr/bin/env python3
"""Coverage-gap analyzer for the StaffML question corpus.

Surfaces where the corpus is thin so generation can be aimed precisely
rather than scattered. Output is a structured report that downstream
batched generation (`gemini_cli_generate_questions.py`) can consume.

The dimensions analyzed:

  1. track            (cloud, edge, mobile, tinyml, global)
  2. competency_area  (13 areas: compute, memory, networking, latency, ...)
  3. zone             (11 ikigai zones)
  4. level            (L1..L6+)
  5. topic            (87 curated topics)
  6. visual coverage  (per archetype in audit_visual_questions.py)

For each axis we compute:
  - count            : how many questions fall here
  - expected_share   : roughly uniform, weighted by track/area importance
  - gap_pct          : (expected - actual) / expected
  - priority         : weighted gap_pct × importance_weight

Top gaps surface as recommended_cells with a target_count_to_fill.

The output report is human-readable Markdown plus machine-readable JSON.
Running this on the current corpus *plus* yesterday's drafts gives a
faithful "what's still missing" view for next-round generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
OUTPUT_DIR = VAULT_DIR / "_validation_results" / "coverage_gaps"

TRACKS = ["cloud", "edge", "mobile", "tinyml", "global"]
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]
ZONES = [
    "recall", "analyze", "design", "implement",
    "diagnosis", "fluency", "specification", "evaluation",
    "realization", "optimization", "mastery",
]
COMPETENCY_AREAS = [
    "compute", "memory", "networking", "latency", "parallelism",
    "deployment", "data", "power", "precision", "reliability",
    "optimization", "architecture", "cross-cutting",
]
VISUAL_ARCHETYPE_TOPICS = {
    "collective-communication", "pipeline-parallelism", "kv-cache-management",
    "queueing-theory", "data-pipeline-engineering", "memory-hierarchy-design",
    "interconnect-topology", "network-bandwidth-bottlenecks", "duty-cycling",
    "fault-tolerance-checkpointing",
}

# Importance weights — drive priority ranking.
TRACK_WEIGHT = {"global": 1.5, "tinyml": 1.3, "mobile": 1.2, "edge": 1.1, "cloud": 0.7}
ZONE_WEIGHT = {
    "realization": 1.5, "specification": 1.3, "mastery": 1.3,
    "evaluation": 1.1, "diagnosis": 1.0, "design": 0.9,
    "fluency": 0.9, "optimization": 0.9, "analyze": 0.8,
    "implement": 0.8, "recall": 0.6,
}
LEVEL_WEIGHT = {"L1": 0.6, "L2": 0.7, "L3": 0.9, "L4": 1.1, "L5": 1.3, "L6+": 1.4}
AREA_WEIGHT = {a: 1.0 for a in COMPETENCY_AREAS}
AREA_WEIGHT.update({"reliability": 1.2, "power": 1.2, "data": 1.1})

# Track × topic blocklist — TinyML can't hold KV-cache, etc.
TRACK_TOPIC_BLOCKLIST = {
    "tinyml": {"kv-cache-management", "pipeline-parallelism",
               "interconnect-topology", "data-pipeline-engineering",
               "distributed-training-economics"},
    "mobile": {"pipeline-parallelism", "interconnect-topology"},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpus(include_drafts: bool = True) -> list[dict[str, Any]]:
    """Read all YAMLs under questions/ and return parsed records."""
    out = []
    for path in QUESTIONS_DIR.glob("**/*.yaml"):
        try:
            d = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d:
            continue
        if not include_drafts and d.get("status") != "published":
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def by_track(records):
    return Counter(r.get("track", "?") for r in records)


def by_track_area(records):
    out = Counter()
    for r in records:
        out[(r.get("track", "?"), r.get("competency_area", "?"))] += 1
    return out


def by_track_zone(records):
    out = Counter()
    for r in records:
        out[(r.get("track", "?"), r.get("zone", "?"))] += 1
    return out


def by_track_level(records):
    out = Counter()
    for r in records:
        out[(r.get("track", "?"), r.get("level", "?"))] += 1
    return out


def by_track_topic(records):
    out = Counter()
    for r in records:
        out[(r.get("track", "?"), r.get("topic", "?"))] += 1
    return out


def visual_coverage(records):
    """For each topic in the archetype catalog, count visual questions."""
    out = {t: 0 for t in VISUAL_ARCHETYPE_TOPICS}
    for r in records:
        if "visual" in r and isinstance(r["visual"], dict):
            topic = r.get("topic", "")
            if topic in out:
                out[topic] += 1
    return out


# ---------------------------------------------------------------------------
# Gap scoring
# ---------------------------------------------------------------------------

def expected_uniform(total: int, n_cells: int) -> float:
    return total / max(n_cells, 1)


def gap_score(actual: int, expected: float, weight: float) -> float:
    """Higher score = bigger gap, more important. Negative score means over-filled."""
    if expected <= 0:
        return 0.0
    return weight * max(0, expected - actual) / expected


def rank_track_area_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every (track, area) cell and return ranked by priority."""
    counts = by_track_area(records)
    track_totals = by_track(records)
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        for area in COMPETENCY_AREAS:
            actual = counts.get((track, area), 0)
            # Expected: track's share of the pie spread across 13 areas
            expected = expected_uniform(track_totals.get(track, 0),
                                        len(COMPETENCY_AREAS))
            w = TRACK_WEIGHT.get(track, 1.0) * AREA_WEIGHT.get(area, 1.0)
            score = gap_score(actual, expected, w)
            rows.append({
                "track": track, "area": area, "actual": actual,
                "expected": round(expected, 1), "weight": round(w, 2),
                "priority": round(score, 2),
            })
    rows.sort(key=lambda r: -r["priority"])
    return rows


def rank_track_zone_level_gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every (track, zone, level) cell."""
    counts = Counter()
    for r in records:
        counts[(r.get("track"), r.get("zone"), r.get("level"))] += 1
    track_totals = by_track(records)
    rows: list[dict[str, Any]] = []
    n_combos_per_track = len(ZONES) * len(LEVELS)
    for track in TRACKS:
        for zone in ZONES:
            for level in LEVELS:
                actual = counts.get((track, zone, level), 0)
                expected = expected_uniform(track_totals.get(track, 0),
                                            n_combos_per_track)
                w = (TRACK_WEIGHT.get(track, 1.0)
                     * ZONE_WEIGHT.get(zone, 1.0)
                     * LEVEL_WEIGHT.get(level, 1.0))
                score = gap_score(actual, expected, w)
                rows.append({
                    "track": track, "zone": zone, "level": level,
                    "actual": actual, "expected": round(expected, 1),
                    "weight": round(w, 2), "priority": round(score, 2),
                })
    rows.sort(key=lambda r: -r["priority"])
    return rows


# ---------------------------------------------------------------------------
# Recommendation: cells to fill next
# ---------------------------------------------------------------------------

def recommend_generation_plan(records: list[dict[str, Any]],
                              total: int, want_visual: bool
                              ) -> list[dict[str, Any]]:
    """Pick `total` cells to target, weighted by gap priority."""
    zone_level_gaps = rank_track_zone_level_gaps(records)
    # Map (track, zone, level) → suggested topic for that area, biased toward
    # visual-eligible topics if --visual.
    cells: list[dict[str, Any]] = []
    seen = set()
    for row in zone_level_gaps:
        if len(cells) >= total:
            break
        track = row["track"]
        if track == "global":
            continue  # global track is cross-track conceptual; harder to target
        # Pick a topic for this (track, zone, level) — prefer visual archetype
        # topics if --visual, otherwise any sensible topic.
        topic_pool = []
        if want_visual:
            topic_pool = [t for t in VISUAL_ARCHETYPE_TOPICS
                          if t not in TRACK_TOPIC_BLOCKLIST.get(track, set())]
        if not topic_pool:
            # Fall back to a pool of broadly-applicable topics
            topic_pool = ["queueing-theory", "memory-hierarchy-design",
                          "compute-cost-estimation", "data-pipeline-engineering",
                          "fault-tolerance-checkpointing",
                          "communication-computation-overlap",
                          "model-serving-infrastructure",
                          "duty-cycling", "quantization-fundamentals"]
            topic_pool = [t for t in topic_pool
                          if t not in TRACK_TOPIC_BLOCKLIST.get(track, set())]
        # Rotate within the pool to diversify topics across cells
        topic = topic_pool[len(cells) % len(topic_pool)]
        key = (track, topic, row["zone"], row["level"])
        if key in seen:
            continue
        seen.add(key)
        cells.append({
            "track": track, "topic": topic, "zone": row["zone"],
            "level": row["level"], "with_visual": (
                want_visual and topic in VISUAL_ARCHETYPE_TOPICS
            ),
            "priority": row["priority"],
            "current_count": row["actual"],
        })
    return cells


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown(records: list[dict[str, Any]],
                    track_area_rows: list[dict[str, Any]],
                    zone_level_rows: list[dict[str, Any]],
                    visual_topic_counts: dict[str, int],
                    plan: list[dict[str, Any]]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(records)
    drafts = sum(1 for r in records if r.get("status") == "draft")
    pub = sum(1 for r in records if r.get("status") == "published")
    visual_count = sum(1 for r in records if "visual" in r)
    visual_pub = sum(1 for r in records
                     if "visual" in r and r.get("status") == "published")

    md = []
    md.append(f"# Coverage Gap Report\n\n*Generated: {today}*\n")
    md.append(f"**Corpus state**: {total} questions ({pub} published, "
              f"{drafts} draft). **Visual**: {visual_count} total, "
              f"{visual_pub} published.\n")
    md.append("\n---\n")

    md.append("\n## Track distribution\n\n| Track | Count |\n|---|---:|")
    track_counts = by_track(records)
    for t in TRACKS:
        md.append(f"| {t} | {track_counts.get(t, 0)} |")

    md.append("\n## Top 20 weakest (track × competency area) cells\n")
    md.append("| Track | Area | Actual | Expected | Weight | Priority |\n"
              "|---|---|---:|---:|---:|---:|")
    for row in track_area_rows[:20]:
        md.append(f"| {row['track']} | {row['area']} | {row['actual']} | "
                  f"{row['expected']} | {row['weight']} | {row['priority']} |")

    md.append("\n## Top 20 weakest (track × zone × level) cells\n")
    md.append("| Track | Zone | Level | Actual | Expected | Weight | Priority |\n"
              "|---|---|---|---:|---:|---:|---:|")
    for row in zone_level_rows[:20]:
        md.append(f"| {row['track']} | {row['zone']} | {row['level']} | "
                  f"{row['actual']} | {row['expected']} | {row['weight']} | "
                  f"{row['priority']} |")

    md.append("\n## Visual archetype coverage\n")
    md.append("| Topic | Count | Status |\n|---|---:|:---:|")
    for t, c in sorted(visual_topic_counts.items(), key=lambda x: x[1]):
        status = "✓" if c >= 1 else "—"
        md.append(f"| {t} | {c} | {status} |")

    md.append(f"\n## Recommended generation plan ({len(plan)} cells)\n")
    md.append("| # | Track | Topic | Zone | Level | Visual? | Priority |\n"
              "|---:|---|---|---|---|:---:|---:|")
    for i, c in enumerate(plan):
        md.append(f"| {i+1} | {c['track']} | {c['topic']} | {c['zone']} | "
                  f"{c['level']} | {'✓' if c['with_visual'] else '—'} | "
                  f"{c['priority']} |")

    md.append("\n---\n*Run `gemini_cli_generate_questions.py --auto-balance "
              "--total N` to fill these cells with batched API calls.*")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-drafts", action="store_true", default=True,
                        help="Include status:draft in counts (default: true).")
    parser.add_argument("--published-only", action="store_true",
                        help="Restrict to status:published (overrides --include-drafts).")
    parser.add_argument("--total", type=int, default=60,
                        help="How many cells to recommend in the plan.")
    parser.add_argument("--visual", action="store_true",
                        help="Bias the plan toward visual-archetype topics.")
    parser.add_argument("--include-areas", default="",
                        help="Comma-separated list of competency_areas (e.g. "
                             "'parallelism,networking') to inject as forced "
                             "targets into the plan. For each listed area, "
                             "add 1 cell per (track, parallelism-flavored "
                             "topic) where the track×area gap is high. "
                             "Closes the structural mismatch where "
                             "topic-level priority misses area-level gaps.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    include_drafts = args.include_drafts and not args.published_only
    records = load_corpus(include_drafts=include_drafts)
    print(f"Loaded {len(records)} records (drafts={'yes' if include_drafts else 'no'})")

    track_area = rank_track_area_gaps(records)
    zone_level = rank_track_zone_level_gaps(records)
    visual_topics = visual_coverage(records)
    plan = recommend_generation_plan(records, args.total, args.visual)

    # E.3: --include-areas. For each listed area, inject hand-targeted
    # cells using area-canonical topics, biased toward (track, area)
    # gaps where the analyzer's topic-priority ranking misses the
    # area-level signal. Each forced cell stamps `forced_area=True` so
    # callers can spot which entries came from the override.
    AREA_TO_TOPICS = {
        "parallelism":  ["pipeline-parallelism", "collective-communication",
                         "kv-cache-management", "interconnect-topology"],
        "networking":   ["network-bandwidth-bottlenecks",
                         "collective-communication", "interconnect-topology"],
        "memory":       ["memory-hierarchy-design", "kv-cache-management"],
        "latency":      ["queueing-theory"],
        "compute":      ["compute-cost-estimation"],
        "data":         ["data-pipeline-engineering"],
        "power":        ["duty-cycling"],
        "precision":    ["quantization-fundamentals"],
        "reliability":  ["fault-tolerance-checkpointing"],
        "deployment":   ["model-serving-infrastructure"],
        "optimization": ["communication-computation-overlap"],
        "architecture": ["kv-cache-management"],
    }
    include_areas = [a.strip() for a in args.include_areas.split(",") if a.strip()]
    if include_areas:
        # For each listed area, walk the track×area ranking and
        # inject up to 6 cells per track (across topics × {L4,L5,L6+}).
        ta_by_priority = sorted(track_area, key=lambda r: -r.get("priority", 0))
        for ta in ta_by_priority:
            area = ta.get("area")
            track = ta.get("track")
            if area not in include_areas:
                continue
            if track == "global" or track == "?":
                continue
            topics = AREA_TO_TOPICS.get(area, [])
            for topic in topics:
                if topic in TRACK_TOPIC_BLOCKLIST.get(track, set()):
                    continue
                for level, zone in [("L4", "diagnosis"), ("L5", "specification"),
                                    ("L6+", "mastery")]:
                    plan.append({
                        "track": track, "topic": topic, "zone": zone,
                        "level": level, "with_visual": False,
                        "priority": round(ta.get("priority", 0) + 0.5, 2),
                        "current_count": 0,
                        "forced_area": area,
                    })
        # Dedup by (track, topic, zone, level) and trim back to total
        seen_keys = set()
        deduped = []
        for c in plan:
            key = (c["track"], c["topic"], c["zone"], c["level"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(c)
        plan = sorted(deduped, key=lambda r: -r.get("priority", 0))[:args.total]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    md = render_markdown(records, track_area, zone_level, visual_topics, plan)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    json_blob = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "include_drafts": include_drafts,
        "track_area_gaps": track_area,
        "track_zone_level_gaps": zone_level,
        "visual_topic_counts": visual_topics,
        "recommended_plan": plan,
    }
    (out_dir / "report.json").write_text(json.dumps(json_blob, indent=2),
                                          encoding="utf-8")

    print(f"\nReport written to {out_dir}/report.md")
    print(f"JSON written to   {out_dir}/report.json")
    print(f"\nTop 5 (track × area) gaps:")
    for r in track_area[:5]:
        print(f"  {r['track']}/{r['area']}: actual={r['actual']} "
              f"expected={r['expected']} priority={r['priority']}")
    print(f"\nTop 5 (track × zone × level) gaps:")
    for r in zone_level[:5]:
        print(f"  {r['track']}/{r['zone']}/{r['level']}: actual={r['actual']} "
              f"expected={r['expected']} priority={r['priority']}")
    print(f"\nVisual archetype coverage:")
    for t, c in sorted(visual_topics.items(), key=lambda x: x[1]):
        print(f"  {t}: {c}")
    print(f"\nRecommended cells to fill: {len(plan)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
