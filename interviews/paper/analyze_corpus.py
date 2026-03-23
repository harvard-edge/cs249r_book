#!/usr/bin/env python3
"""Analyze corpus.json and write structured stats to corpus_stats.json.

This is the "analyze" step in the pipeline:
  corpus.json → analyze_corpus.py → corpus_stats.json → generate_figures.py → PDFs

Run: python3 analyze_corpus.py
Reads: ../corpus.json, ../chains.json
Writes: corpus_stats.json
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).parent
CORPUS_PATH = PAPER_DIR.parent / "corpus.json"
CHAINS_PATH = PAPER_DIR.parent / "chains.json"
OUTPUT_PATH = PAPER_DIR / "corpus_stats.json"

TRACKS = ["cloud", "edge", "mobile", "tinyml"]
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]
BLOOM_LABELS = {
    "L1": "Remember", "L2": "Understand", "L3": "Apply",
    "L4": "Analyze", "L5": "Evaluate", "L6+": "Create",
}


def classify_format(scenario: str) -> list[str]:
    s = scenario.lower()
    fmts = []
    if any(w in s for w in ["calculate", "compute", "estimate", "how many", "how much"]):
        fmts.append("calculation")
    if any(w in s for w in ["design", "architect", "propose", "how would you build"]):
        fmts.append("design")
    if any(w in s for w in ["explain", "what is", "define", "describe"]):
        fmts.append("conceptual")
    if any(w in s for w in ["optimize", "improve", "reduce", "speed up"]):
        fmts.append("optimization")
    if any(w in s for w in ["diagnose", "debug", "why is", "root cause", "fails"]):
        fmts.append("diagnosis")
    if any(w in s for w in ["compare", "trade-off", "tradeoff", "versus", " vs "]):
        fmts.append("tradeoff")
    return fmts if fmts else ["conceptual"]


def main():
    corpus = json.loads(CORPUS_PATH.read_text())
    chains = json.loads(CHAINS_PATH.read_text()) if CHAINS_PATH.exists() else []

    all_qs = corpus
    pub = [q for q in corpus if q.get("status", "published") == "published"]
    archived = [q for q in corpus if q.get("status") == "archived"]

    stats = {}

    # ── Summary ──
    stats["summary"] = {
        "total": len(all_qs),
        "published": len(pub),
        "archived": len(archived),
        "tracks": len(TRACKS),
        "levels": len(LEVELS),
        "chains_total": len(chains),
        "chains_full": sum(1 for ch in chains if len(ch.get("levels", [])) >= 6),
    }

    # ── Track × Level matrix ──
    matrix = defaultdict(lambda: defaultdict(int))
    for q in pub:
        if q["track"] in TRACKS:
            matrix[q["track"]][q["level"]] += 1

    stats["track_level_matrix"] = {
        "tracks": TRACKS,
        "levels": LEVELS,
        "data": {t: {l: matrix[t][l] for l in LEVELS} for t in TRACKS},
        "track_totals": {t: sum(matrix[t][l] for l in LEVELS) for t in TRACKS},
    }

    # ── Competency areas ──
    areas = Counter(q["competency_area"] for q in pub)
    stats["competency_areas"] = {a: c for a, c in areas.most_common()}

    # ── Field coverage ──
    total = len(pub)
    stats["field_coverage"] = {
        "competency_area": sum(1 for q in pub if q.get("competency_area", "").strip()) / total,
        "napkin_math": sum(1 for q in pub if q.get("details", {}).get("napkin_math", "").strip()) / total,
        "common_mistake": sum(1 for q in pub if q.get("details", {}).get("common_mistake", "").strip()) / total,
        "realistic_solution": sum(1 for q in pub if q.get("details", {}).get("realistic_solution", "").strip()) / total,
        "deep_dive_url": sum(1 for q in pub if q.get("details", {}).get("deep_dive_url", "").strip()) / total,
        "bloom_level": sum(1 for q in pub if q.get("bloom_level", "").strip()) / total,
        "canonical_topic": sum(1 for q in pub if q.get("canonical_topic", "").strip()) / total,
        "mcq_options": sum(1 for q in pub if q.get("details", {}).get("options")) / total,
    }

    # ── Format distribution by level ──
    formats = ["calculation", "design", "conceptual", "optimization", "diagnosis", "tradeoff"]
    format_by_level = {}
    for level in LEVELS:
        lqs = [q for q in pub if q["level"] == level]
        fmt_counts = Counter()
        for q in lqs:
            for fmt in classify_format(q["scenario"]):
                fmt_counts[fmt] += 1
        fmt_total = sum(fmt_counts.values()) or 1
        format_by_level[level] = {
            "total_questions": len(lqs),
            "format_pct": {fmt: round(100 * fmt_counts.get(fmt, 0) / fmt_total, 1) for fmt in formats},
            "format_counts": {fmt: fmt_counts.get(fmt, 0) for fmt in formats},
        }
    stats["format_by_level"] = format_by_level

    # ── Coverage cube ──
    cube = defaultdict(int)
    area_list = sorted(set(q["competency_area"] for q in pub if q["competency_area"]))
    for q in pub:
        if q["track"] in TRACKS and q.get("competency_area"):
            cube[(q["track"], q["level"], q["competency_area"])] += 1

    empty = sum(1 for t in TRACKS for l in LEVELS for a in area_list if cube.get((t, l, a), 0) == 0)
    underfilled = sum(1 for t in TRACKS for l in LEVELS for a in area_list if 0 < cube.get((t, l, a), 0) < 3)
    healthy = sum(1 for t in TRACKS for l in LEVELS for a in area_list if cube.get((t, l, a), 0) >= 3)

    stats["coverage_cube"] = {
        "empty_cells": empty,
        "underfilled_cells": underfilled,
        "healthy_cells": healthy,
        "total_cells": empty + underfilled + healthy,
    }

    # ── Level distribution percentages ──
    level_dist = {}
    for t in TRACKS:
        t_total = sum(matrix[t][l] for l in LEVELS) or 1
        level_dist[t] = {l: round(100 * matrix[t][l] / t_total) for l in LEVELS}
    stats["level_distribution_pct"] = level_dist

    # ── Error rates ──
    stats["error_rates"] = {
        "round1_raw": 4.3,
        "round1_confirmed": 1.5,
        "round2_confirmed": 0.22,
        "round1_errors": 48,
        "round1_false_positives": 10,
        "round2_errors": 7,
    }

    # ── Dedup stats ──
    stats["dedup"] = {
        "exact_dupes": 0,
        "fuzzy_dupes": 0,
        "semantic_flagged": 502,
        "archived": len(archived),
    }

    # ── Chain stats ──
    chain_levels = Counter()
    for ch in chains:
        n = len(ch.get("levels", ch.get("questions", [])))
        chain_levels[n] += 1

    stats["chains"] = {
        "total": len(chains),
        "by_span": {str(k): v for k, v in sorted(chain_levels.items())},
        "full_chains_6_levels": sum(1 for ch in chains if len(ch.get("levels", [])) >= 6),
        "missing_foundation": sum(
            1 for ch in chains
            if set(ch.get("levels", [])) & {"L5", "L6+"}
            and not set(ch.get("levels", [])) & {"L1", "L2"}
        ),
        "missing_depth": sum(
            1 for ch in chains
            if set(ch.get("levels", [])) & {"L1", "L2"}
            and not set(ch.get("levels", [])) & {"L4", "L5", "L6+"}
        ),
    }

    # ── Taxonomy stats ──
    topics = set(q.get("canonical_topic", q["topic"]) for q in pub)
    singleton_topics = sum(
        1 for t in topics
        if sum(1 for q in pub if q.get("canonical_topic", q["topic"]) == t) == 1
    )
    stats["taxonomy"] = {
        "raw_topics": len(set(q["topic"] for q in pub)),
        "canonical_topics": len(topics),
        "singleton_topics": singleton_topics,
    }

    # ── Depth chain example (KV-cache) ──
    kv_chain = next((ch for ch in chains if "kv-cache" in ch.get("topic", "")), None)
    if kv_chain:
        by_id = {q["id"]: q for q in corpus}
        stats["example_chain"] = {
            "topic": kv_chain["topic"],
            "competency_area": kv_chain.get("competency_area", "memory"),
            "questions": [
                {
                    "level": cq["level"],
                    "bloom": BLOOM_LABELS.get(cq["level"], ""),
                    "title": cq["title"],
                    "scenario_preview": by_id.get(cq["id"], {}).get("scenario", "")[:100],
                }
                for cq in kv_chain["questions"]
            ],
        }

    # ── Write ──
    OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({len(stats)} sections)")
    print(f"  Published: {stats['summary']['published']}")
    print(f"  Archived: {stats['summary']['archived']}")
    print(f"  Chains: {stats['summary']['chains_total']} ({stats['summary']['chains_full']} full)")
    print(f"  Coverage: {stats['coverage_cube']['healthy_cells']} healthy, "
          f"{stats['coverage_cube']['underfilled_cells']} underfilled, "
          f"{stats['coverage_cube']['empty_cells']} empty")


if __name__ == "__main__":
    main()
