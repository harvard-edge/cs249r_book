#!/usr/bin/env python3
"""Analyze corpus.json and write structured stats to corpus_stats.json.

This is the "analyze" step in the pipeline:
  vault/corpus.json → analyze_corpus.py → corpus_stats.json → generate_figures.py → PDFs

Run: python3 analyze_corpus.py
Reads: ../vault/corpus.json, ../vault/chains.json, ../vault/schema/taxonomy_data.yaml
Writes: corpus_stats.json
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).parent
PAPER_DIR = SCRIPTS_DIR.parent
VAULT_DIR = PAPER_DIR.parent / "vault"
CORPUS_PATH = VAULT_DIR / "corpus.json"
CHAINS_PATH = VAULT_DIR / "chains.json"
TAXONOMY_PATH = VAULT_DIR / "schema" / "taxonomy_data.yaml"
OUTPUT_PATH = PAPER_DIR / "corpus_stats.json"

TRACKS = ["cloud", "edge", "mobile", "tinyml", "global"]
LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6+"]
ZONES = [
    "recall", "analyze", "design", "implement",
    "diagnosis", "specification", "fluency",
    "evaluation", "realization", "optimization",
    "mastery",
]
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

    # Load taxonomy for graph stats
    taxonomy = None
    if TAXONOMY_PATH.exists():
        taxonomy = yaml.safe_load(TAXONOMY_PATH.read_text())

    pub = [q for q in corpus if q.get("status", "published") == "published"]
    archived = [q for q in corpus if q.get("status") == "archived"]

    stats = {}

    # ── Summary ──────────────────────────────────────────────
    topic_set = set(q.get("topic", "") for q in pub if q.get("topic"))
    zone_set = set(q.get("zone", "") for q in pub if q.get("zone"))
    area_set = set(q.get("competency_area", "") for q in pub if q.get("competency_area"))

    full_chains = sum(
        1 for ch in chains
        if set(ch.get("levels", [])) >= {"L1", "L2", "L3", "L4", "L5", "L6+"}
    )

    stats["summary"] = {
        "total": len(pub),
        "published": len(pub),
        "archived": len(archived),
        "tracks": len(set(q.get("track") for q in pub)),
        "levels": len(LEVELS),
        "topics": len(topic_set),
        "zones": len(zone_set),
        "areas": len(area_set),
        "chains_total": len(chains),
        "chains_full": full_chains,
    }

    # ── Track × Level matrix ─────────────────────────────────
    matrix = defaultdict(lambda: defaultdict(int))
    for q in pub:
        matrix[q["track"]][q["level"]] += 1

    stats["track_level_matrix"] = {
        "tracks": TRACKS,
        "levels": LEVELS,
        "data": {t: {l: matrix[t][l] for l in LEVELS} for t in TRACKS},
        "track_totals": {t: sum(matrix[t][l] for l in LEVELS) for t in TRACKS},
    }

    # ── Competency areas ─────────────────────────────────────
    areas = Counter(q["competency_area"] for q in pub)
    stats["competency_areas"] = {a: c for a, c in areas.most_common()}

    # ── Zone distribution ────────────────────────────────────
    zone_counts = Counter(q.get("zone", "") for q in pub)
    stats["zone_distribution"] = {z: zone_counts.get(z, 0) for z in ZONES}

    # ── Zone × Level matrix ──────────────────────────────────
    zl_matrix = defaultdict(lambda: defaultdict(int))
    for q in pub:
        zl_matrix[q.get("zone", "")][q["level"]] += 1
    stats["zone_level_matrix"] = {
        z: {l: zl_matrix[z][l] for l in LEVELS} for z in ZONES
    }

    # ── Zone × Track matrix ──────────────────────────────────
    zt_matrix = defaultdict(lambda: defaultdict(int))
    for q in pub:
        zt_matrix[q.get("zone", "")][q["track"]] += 1
    stats["zone_track_matrix"] = {
        z: {t: zt_matrix[z][t] for t in TRACKS} for z in ZONES
    }

    # ── Topic distribution ───────────────────────────────────
    topic_counts = Counter(q.get("topic", "") for q in pub)
    stats["topic_distribution"] = {t: c for t, c in topic_counts.most_common()}

    # ── Bloom distribution ───────────────────────────────────
    bloom_counts = Counter(q.get("bloom_level", "") for q in pub)
    stats["bloom_distribution"] = dict(bloom_counts.most_common())

    # ── Field coverage ───────────────────────────────────────
    total = len(pub)
    stats["field_coverage"] = {
        "topic": sum(1 for q in pub if q.get("topic", "").strip()) / total,
        "zone": sum(1 for q in pub if q.get("zone", "").strip()) / total,
        "competency_area": sum(1 for q in pub if q.get("competency_area", "").strip()) / total,
        "napkin_math": sum(1 for q in pub if q.get("details", {}).get("napkin_math", "").strip()) / total,
        "common_mistake": sum(1 for q in pub if q.get("details", {}).get("common_mistake", "").strip()) / total,
        "realistic_solution": sum(1 for q in pub if q.get("details", {}).get("realistic_solution", "").strip()) / total,
        "bloom_level": sum(1 for q in pub if q.get("bloom_level", "").strip()) / total,
    }

    # ── Format distribution by level ─────────────────────────
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

    # ── Coverage cube (track × level × area) ─────────────────
    cube = defaultdict(int)
    area_list = sorted(set(q["competency_area"] for q in pub if q["competency_area"]))
    for q in pub:
        if q.get("competency_area"):
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

    # ── Level distribution percentages ───────────────────────
    level_dist = {}
    for t in TRACKS:
        t_total = sum(matrix[t][l] for l in LEVELS) or 1
        level_dist[t] = {l: round(100 * matrix[t][l] / t_total) for l in LEVELS}
    stats["level_distribution_pct"] = level_dist

    # ── Validation stats (computed from data, not hardcoded) ─
    validated_true = sum(1 for q in pub if q.get("validated") is True)
    validated_false = sum(1 for q in pub if q.get("validated") is False)
    has_issues = sum(1 for q in pub if q.get("validation_issues"))

    stats["validation"] = {
        "validated_true": validated_true,
        "validated_false": validated_false,
        "validated_null": total - validated_true - validated_false,
        "has_issues": has_issues,
        "validated_pct": round(100 * validated_true / total, 1),
    }

    # ── Chain stats ──────────────────────────────────────────
    chain_levels = Counter()
    for ch in chains:
        n = len(ch.get("questions", []))
        chain_levels[n] += 1

    chained_qids = set()
    for ch in chains:
        for entry in ch.get("questions", []):
            qid = entry["id"] if isinstance(entry, dict) else entry
            chained_qids.add(qid)

    stats["chains"] = {
        "total": len(chains),
        "by_length": {str(k): v for k, v in sorted(chain_levels.items())},
        "full_chains": full_chains,
        "questions_in_chains": len(chained_qids),
        "chain_coverage_pct": round(100 * len(chained_qids) / total, 1),
    }

    # ── Taxonomy graph stats ─────────────────────────────────
    if taxonomy:
        topics_data = taxonomy.get("topics", [])
        edge_types = Counter()
        for t in topics_data:
            for e in t.get("edges", []):
                edge_types[e["edge_type"]] += 1

        # Compute prerequisite depth
        prereq_adj = defaultdict(list)
        for t in topics_data:
            for e in t.get("edges", []):
                if e["edge_type"] == "prerequisite":
                    prereq_adj[t["id"]].append(e["target"])

        def depth(tid, visited=None):
            if visited is None:
                visited = set()
            if tid in visited:
                return 0
            visited.add(tid)
            children = prereq_adj.get(tid, [])
            if not children:
                return 0
            return 1 + max(depth(c, visited) for c in children)

        max_depth = max((depth(t["id"]) for t in topics_data), default=0)

        # Root topics (no prerequisites)
        has_prereqs = set()
        for t in topics_data:
            for e in t.get("edges", []):
                if e["edge_type"] == "prerequisite":
                    has_prereqs.add(t["id"])

        stats["taxonomy_graph"] = {
            "total_topics": len(topics_data),
            "total_edges": sum(edge_types.values()),
            "by_type": dict(edge_types),
            "max_prerequisite_depth": max_depth,
            "root_topics": len(topics_data) - len(has_prereqs),
            "topics_per_area": dict(Counter(t["area"] for t in topics_data).most_common()),
        }

    # ── Cross-track topic coverage ───────────────────────────
    topic_tracks = defaultdict(set)
    for q in pub:
        if q.get("topic"):
            topic_tracks[q["topic"]].add(q["track"])
    stats["cross_track_coverage"] = {
        t: sorted(tracks) for t, tracks in sorted(topic_tracks.items())
    }

    # ── Depth chain example (KV-cache) ───────────────────────
    kv_chain = next((ch for ch in chains if "kv-cache" in ch.get("topic", "")), None)
    if kv_chain:
        by_id = {q["id"]: q for q in corpus}
        stats["example_chain"] = {
            "topic": kv_chain["topic"],
            "competency_area": kv_chain.get("competency_area", "memory"),
            "questions": [
                {
                    "level": cq.get("level", by_id.get(cq.get("id", cq if isinstance(cq, str) else ""), {}).get("level", "?")),
                    "bloom": cq.get("bloom", ""),
                    "title": cq.get("title", by_id.get(cq.get("id", ""), {}).get("title", "?")),
                    "scenario_preview": by_id.get(
                        cq["id"] if isinstance(cq, dict) else cq, {}
                    ).get("scenario", "")[:100],
                }
                for cq in kv_chain["questions"]
            ],
        }

    # ── Write ────────────────────────────────────────────────
    OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
    s = stats["summary"]
    print(f"Wrote {OUTPUT_PATH} ({len(stats)} sections)")
    print(f"  Questions: {s['total']}")
    print(f"  Topics: {s['topics']}, Zones: {s['zones']}, Areas: {s['areas']}")
    print(f"  Chains: {s['chains_total']} ({s['chains_full']} full L1→L6+)")
    print(f"  Validated: {stats['validation']['validated_pct']}%")
    cc = stats["coverage_cube"]
    print(f"  Coverage: {cc['healthy_cells']} healthy, "
          f"{cc['underfilled_cells']} underfilled, {cc['empty_cells']} empty")


if __name__ == "__main__":
    main()
