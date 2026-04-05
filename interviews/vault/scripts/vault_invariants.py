#!/usr/bin/env python3
"""Vault Invariant Checks — structural guardrails for StaffML data integrity.

This script enforces invariants ACROSS the three data files (corpus.json,
taxonomy.json, chains.json) that per-question schema validation cannot catch.
Run it as a pre-commit gate or CI check whenever vault data changes.

Usage:
    python3 vault_invariants.py              # Run all checks, exit 1 on any FAIL
    python3 vault_invariants.py --fix        # Auto-fix what can be fixed, report rest
    python3 vault_invariants.py --check N    # Run only check N (e.g., --check 3)
    python3 vault_invariants.py --json       # Output results as JSON

Checks:
    1. No duplicate concept names in taxonomy
    2. No duplicate concept IDs in taxonomy
    3. All concept IDs are kebab-case (no Title Case stubs)
    4. taxonomy.question_count matches actual corpus primary_concept counts
    5. All corpus primary_concepts exist in taxonomy
    6. All taxonomy prerequisites exist as defined concepts (no orphans)
    7. No cycles in prerequisite graph
    8. competency_area uses only canonical values
    9. level uses only canonical values (no bare "L6")
   10. No duplicate question IDs in corpus
   11. All chain question IDs exist in corpus
   12. Title uniqueness within (track, level) — warn only
   13. Singleton taxonomy concepts (no prereqs AND never a prereq) — warn only
   14. Corpus concepts not in taxonomy — warn only

Exit codes:
    0 = all checks pass (warnings OK)
    1 = one or more FAIL checks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Resolve paths relative to this script's parent (vault/)
VAULT = Path(__file__).resolve().parent.parent
CORPUS_PATH = VAULT / "corpus.json"
TAXONOMY_PATH = VAULT / "taxonomy.json"
CHAINS_PATH = VAULT / "chains.json"

# Canonical controlled vocabularies (mirror schema.py)
CANONICAL_AREAS = {
    "compute", "memory", "latency", "precision", "power",
    "architecture", "optimization", "parallelism", "networking",
    "deployment", "reliability", "data", "cross-cutting",
}
CANONICAL_LEVELS = {"L1", "L2", "L3", "L4", "L5", "L6+"}
CANONICAL_TRACKS = {"cloud", "edge", "mobile", "tinyml", "global"}

# ── Results ──────────────────────────────────────────────────

class CheckResult:
    def __init__(self, num: int, name: str, status: str, message: str,
                 details: list[str] | None = None, fixable: bool = False):
        self.num = num
        self.name = name
        self.status = status  # PASS, FAIL, WARN, FIXED
        self.message = message
        self.details = details or []
        self.fixable = fixable

    def to_dict(self):
        return {
            "check": self.num, "name": self.name, "status": self.status,
            "message": self.message, "details": self.details[:20],
            "total_issues": len(self.details),
        }


# ── Data Loading ─────────────────────────────────────────────

def load_data():
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    with open(CHAINS_PATH) as f:
        chains = json.load(f)
    return corpus, taxonomy, chains


# ── Individual Checks ────────────────────────────────────────

def check_01_duplicate_concept_names(taxonomy, **_) -> CheckResult:
    names = Counter(c["name"] for c in taxonomy["concepts"])
    dupes = {n: cnt for n, cnt in names.items() if cnt > 1}
    if not dupes:
        return CheckResult(1, "duplicate_concept_names", "PASS",
                           f"All {len(taxonomy['concepts'])} concept names are unique")
    details = [f'"{n}" appears {cnt} times' for n, cnt in sorted(dupes.items())]
    return CheckResult(1, "duplicate_concept_names", "FAIL",
                       f"{len(dupes)} duplicate concept names", details, fixable=True)


def check_02_duplicate_concept_ids(taxonomy, **_) -> CheckResult:
    ids = Counter(c["id"] for c in taxonomy["concepts"])
    dupes = {i: cnt for i, cnt in ids.items() if cnt > 1}
    if not dupes:
        return CheckResult(2, "duplicate_concept_ids", "PASS",
                           f"All {len(taxonomy['concepts'])} concept IDs are unique")
    details = [f'"{i}" appears {cnt} times' for i, cnt in sorted(dupes.items())]
    return CheckResult(2, "duplicate_concept_ids", "FAIL",
                       f"{len(dupes)} duplicate concept IDs", details)


def check_03_kebab_case_ids(taxonomy, **_) -> CheckResult:
    kebab_re = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
    bad = [c["id"] for c in taxonomy["concepts"] if not kebab_re.match(c["id"])]
    if not bad:
        return CheckResult(3, "kebab_case_ids", "PASS",
                           "All concept IDs are kebab-case")
    details = [f'"{i}" — should be "{i.lower().replace(" ", "-")}"' for i in bad]
    return CheckResult(3, "kebab_case_ids", "FAIL",
                       f"{len(bad)} non-kebab-case concept IDs", details, fixable=True)


def check_04_question_count_sync(taxonomy, corpus, **_) -> CheckResult:
    actual = Counter(q.get("primary_concept", "") for q in corpus)
    mismatches = []
    for c in taxonomy["concepts"]:
        claimed = c.get("question_count", 0)
        real = actual.get(c["id"], 0)
        if claimed != real:
            mismatches.append(f'{c["id"]}: taxonomy says {claimed}, corpus has {real}')
    if not mismatches:
        return CheckResult(4, "question_count_sync", "PASS",
                           "All taxonomy question_counts match corpus")
    return CheckResult(4, "question_count_sync", "FAIL",
                       f"{len(mismatches)} question_count mismatches",
                       mismatches, fixable=True)


def check_05_corpus_concepts_in_taxonomy(taxonomy, corpus, **_) -> CheckResult:
    tax_ids = {c["id"] for c in taxonomy["concepts"]}
    corpus_concepts = {q.get("primary_concept", "") for q in corpus} - {"", None}
    missing = corpus_concepts - tax_ids
    if not missing:
        return CheckResult(5, "corpus_concepts_in_taxonomy", "PASS",
                           "All corpus primary_concepts exist in taxonomy")
    details = sorted(missing)
    return CheckResult(5, "corpus_concepts_in_taxonomy", "WARN",
                       f"{len(missing)} corpus concepts not in taxonomy", details)


def check_06_prerequisite_integrity(taxonomy, **_) -> CheckResult:
    defined = {c["id"] for c in taxonomy["concepts"]}
    orphans = set()
    for c in taxonomy["concepts"]:
        for p in c.get("prerequisites", []):
            if p not in defined:
                orphans.add(f'{c["id"]} requires "{p}" (not defined)')
    if not orphans:
        return CheckResult(6, "prerequisite_integrity", "PASS",
                           "All prerequisites reference defined concepts")
    return CheckResult(6, "prerequisite_integrity", "FAIL",
                       f"{len(orphans)} orphan prerequisites", sorted(orphans))


def check_07_no_cycles(taxonomy, **_) -> CheckResult:
    adj = defaultdict(list)
    for c in taxonomy["concepts"]:
        for p in c.get("prerequisites", []):
            adj[p].append(c["id"])  # p -> c (prereq flows forward)

    all_ids = {c["id"] for c in taxonomy["concepts"]}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in all_ids}
    cycles = []

    def dfs(node, path):
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycles.append(" → ".join(path[cycle_start:] + [neighbor]))
            elif color[neighbor] == WHITE:
                dfs(neighbor, path + [neighbor])
        color[node] = BLACK

    for node in all_ids:
        if color[node] == WHITE:
            dfs(node, [node])

    if not cycles:
        return CheckResult(7, "no_cycles", "PASS", "Prerequisite graph is acyclic (DAG)")
    return CheckResult(7, "no_cycles", "FAIL",
                       f"{len(cycles)} cycles detected", cycles[:10])


def check_08_canonical_competency_area(corpus, **_) -> CheckResult:
    bad = []
    for q in corpus:
        area = q.get("competency_area", "")
        if area not in CANONICAL_AREAS:
            bad.append(f'{q["id"]}: "{area}"')
    if not bad:
        return CheckResult(8, "canonical_competency_area", "PASS",
                           "All competency_area values are canonical")
    areas = Counter(q.get("competency_area", "") for q in corpus
                    if q.get("competency_area", "") not in CANONICAL_AREAS)
    summary = [f'"{a}": {cnt} questions' for a, cnt in areas.most_common(20)]
    return CheckResult(8, "canonical_competency_area", "FAIL",
                       f"{len(bad)} questions with non-canonical competency_area",
                       summary, fixable=True)


def check_09_canonical_levels(corpus, **_) -> CheckResult:
    bad = []
    for q in corpus:
        level = q.get("level", "")
        if level not in CANONICAL_LEVELS:
            bad.append(f'{q["id"]}: "{level}"')
    if not bad:
        return CheckResult(9, "canonical_levels", "PASS",
                           "All level values are canonical")
    return CheckResult(9, "canonical_levels", "FAIL",
                       f"{len(bad)} questions with non-canonical level", bad[:20],
                       fixable=True)


def check_10_duplicate_question_ids(corpus, **_) -> CheckResult:
    ids = Counter(q["id"] for q in corpus)
    dupes = {i: cnt for i, cnt in ids.items() if cnt > 1}
    if not dupes:
        return CheckResult(10, "duplicate_question_ids", "PASS",
                           f"All {len(corpus)} question IDs are unique")
    details = [f'"{i}" appears {cnt} times' for i, cnt in sorted(dupes.items())]
    return CheckResult(10, "duplicate_question_ids", "FAIL",
                       f"{len(dupes)} duplicate question IDs", details)


def check_11_chain_question_ids(corpus, chains, **_) -> CheckResult:
    corpus_ids = {q["id"] for q in corpus}
    missing = []
    for chain in chains:
        for entry in chain.get("questions", []):
            # Questions can be strings (IDs) or dicts with an "id" field
            qid = entry["id"] if isinstance(entry, dict) else entry
            if qid not in corpus_ids:
                missing.append(f'chain {chain.get("chain_id","?")}: "{qid}" not in corpus')
    if not missing:
        return CheckResult(11, "chain_question_ids", "PASS",
                           "All chain question IDs exist in corpus")
    return CheckResult(11, "chain_question_ids", "FAIL",
                       f"{len(missing)} chain references to missing questions",
                       missing[:20])


def check_12_title_uniqueness(corpus, **_) -> CheckResult:
    seen = defaultdict(list)
    for q in corpus:
        key = (q.get("track"), q.get("level"), q.get("title"))
        seen[key].append(q["id"])
    dupes = {k: ids for k, ids in seen.items() if len(ids) > 1}
    if not dupes:
        return CheckResult(12, "title_uniqueness", "PASS",
                           "No duplicate titles within (track, level)")
    details = [f'"{k[2]}" in {k[0]}/{k[1]}: {len(ids)} copies'
               for k, ids in sorted(dupes.items(), key=lambda x: -len(x[1]))[:20]]
    return CheckResult(12, "title_uniqueness", "WARN",
                       f"{len(dupes)} duplicate (track, level, title) combinations",
                       details)


def check_13_singleton_concepts(taxonomy, corpus, **_) -> CheckResult:
    defined = {c["id"] for c in taxonomy["concepts"]}
    all_prereqs = set()
    has_prereqs = set()
    for c in taxonomy["concepts"]:
        prereqs = c.get("prerequisites", [])
        if prereqs:
            has_prereqs.add(c["id"])
        for p in prereqs:
            all_prereqs.add(p)

    # Singletons: no prereqs AND never referenced as prereq
    singletons = defined - has_prereqs - all_prereqs
    # Only warn about singletons that actually have questions (the rest may be prunable)
    actual_counts = Counter(q.get("primary_concept", "") for q in corpus)
    active_singletons = [s for s in singletons if actual_counts.get(s, 0) > 0]

    if not active_singletons:
        return CheckResult(13, "singleton_concepts", "PASS",
                           "No active singleton concepts (all are connected)")
    details = [f'{s}: {actual_counts[s]} questions' for s in
               sorted(active_singletons, key=lambda x: -actual_counts.get(x, 0))[:30]]
    return CheckResult(13, "singleton_concepts", "WARN",
                       f"{len(active_singletons)} active singleton concepts "
                       f"(no prereqs and never a prereq)",
                       details)


def check_14_corpus_only_concepts(taxonomy, corpus, **_) -> CheckResult:
    tax_ids = {c["id"] for c in taxonomy["concepts"]}
    corpus_concepts = Counter(q.get("primary_concept", "") for q in corpus)
    corpus_only = {c: cnt for c, cnt in corpus_concepts.items()
                   if c and c not in tax_ids}
    if not corpus_only:
        return CheckResult(14, "corpus_only_concepts", "PASS",
                           "All corpus concepts are in taxonomy")
    details = [f'{c}: {cnt} questions' for c, cnt in
               sorted(corpus_only.items(), key=lambda x: -x[1])[:30]]
    return CheckResult(14, "corpus_only_concepts", "WARN",
                       f"{len(corpus_only)} corpus concepts not in taxonomy "
                       f"({sum(corpus_only.values())} questions)",
                       details)


# ── Gold Standard Checks (v6.0) ──────────────────────────────

VALID_ZONES = {
    "recall", "analyze", "design", "implement", "diagnosis",
    "specification", "fluency", "evaluation", "realization",
    "optimization", "mastery",
}

VALID_TOPICS = {
    "roofline-analysis", "gpu-compute-architecture", "accelerator-comparison",
    "mcu-compute-constraints", "systolic-dataflow", "compute-cost-estimation",
    "vram-budgeting", "kv-cache-management", "memory-hierarchy-design",
    "activation-memory", "memory-mapped-inference", "tensor-arena-planning",
    "dma-data-movement", "memory-pressure-management",
    "latency-decomposition", "batching-strategies", "tail-latency",
    "real-time-deadlines", "profiling-bottleneck-analysis", "queueing-theory",
    "quantization-fundamentals", "mixed-precision-training", "extreme-quantization",
    "power-budgeting", "thermal-management", "energy-per-operation",
    "duty-cycling", "datacenter-efficiency",
    "transformer-systems-cost", "cnn-efficient-design", "attention-scaling",
    "mixture-of-experts", "model-size-estimation", "neural-architecture-search",
    "encoder-decoder-tradeoffs",
    "pruning-sparsity", "knowledge-distillation", "kernel-fusion",
    "graph-compilation", "operator-scheduling", "flash-attention",
    "speculative-decoding",
    "data-parallelism", "model-tensor-parallelism", "pipeline-parallelism",
    "3d-parallelism", "gradient-synchronization", "scheduling-resource-management",
    "collective-communication", "interconnect-topology",
    "network-bandwidth-bottlenecks", "rdma-transport", "load-balancing",
    "congestion-control",
    "model-serving-infrastructure", "mlops-lifecycle", "ota-firmware-updates",
    "container-orchestration", "model-format-conversion", "ab-rollout-strategies",
    "compound-ai-systems",
    "fault-tolerance-checkpointing", "distribution-drift-detection",
    "graceful-degradation", "safety-certification", "adversarial-robustness",
    "monitoring-observability",
    "data-pipeline-engineering", "feature-store-management",
    "data-quality-validation", "dataset-curation", "streaming-ingestion",
    "storage-format-selection", "data-efficiency-selection",
    "federated-learning", "differential-privacy", "fairness-evaluation",
    "responsible-ai", "tco-cost-modeling",
}


def check_15_zone_coverage(corpus, **_) -> CheckResult:
    zones = {q.get("zone") for q in corpus if q.get("zone")}
    missing = VALID_ZONES - zones
    if len(zones) >= 9:
        if missing:
            return CheckResult(15, "zone_coverage", "WARN",
                               f"{len(zones)}/11 zones populated (missing: {sorted(missing)})")
        return CheckResult(15, "zone_coverage", "PASS",
                           f"All 11 zones populated")
    return CheckResult(15, "zone_coverage", "FAIL",
                       f"Only {len(zones)}/11 zones populated",
                       [f"Missing: {sorted(missing)}"])


def check_16_topic_coverage(corpus, **_) -> CheckResult:
    topic_counts = Counter(q.get("topic", "") for q in corpus)
    zero_topics = sorted(t for t in VALID_TOPICS if topic_counts.get(t, 0) == 0)
    if not zero_topics:
        return CheckResult(16, "topic_coverage", "PASS",
                           f"All {len(VALID_TOPICS)} topics have questions")
    details = [f"{t}: 0 questions" for t in zero_topics]
    return CheckResult(16, "topic_coverage", "WARN",
                       f"{len(zero_topics)} topics with 0 questions", details)


def check_17_topic_concentration(corpus, **_) -> CheckResult:
    topic_counts = Counter(q.get("topic", "") for q in corpus)
    threshold = len(corpus) * 0.15
    overloaded = [(t, cnt) for t, cnt in topic_counts.most_common()
                  if cnt > threshold]
    if not overloaded:
        return CheckResult(17, "topic_concentration", "PASS",
                           f"No topic exceeds 15% of corpus")
    details = [f"{t}: {cnt} questions ({100*cnt/len(corpus):.1f}%)"
               for t, cnt in overloaded]
    return CheckResult(17, "topic_concentration", "WARN",
                       f"{len(overloaded)} topics exceed 15% of corpus", details)


def check_18_chain_levels_populated(chains, **_) -> CheckResult:
    null_levels = [c.get("chain_id", "?") for c in chains
                   if not c.get("levels")]
    if not null_levels:
        return CheckResult(18, "chain_levels_populated", "PASS",
                           "All chains have levels populated")
    return CheckResult(18, "chain_levels_populated", "FAIL",
                       f"{len(null_levels)} chains with null levels",
                       null_levels[:20])


def check_19_validated_consistency(corpus, **_) -> CheckResult:
    inconsistent = []
    for q in corpus:
        vs = q.get("validation_status")
        v = q.get("validated")
        issues = q.get("validation_issues")
        if vs == "OK" and v is False and (not issues or issues == []):
            inconsistent.append(q["id"])
    if not inconsistent:
        return CheckResult(19, "validated_consistency", "PASS",
                           "validation_status and validated are consistent")
    return CheckResult(19, "validated_consistency", "FAIL",
                       f"{len(inconsistent)} questions: status=OK but validated=false with no issues",
                       inconsistent[:20], fixable=True)


# ── Auto-Fix Functions ───────────────────────────────────────

def fix_04_question_count_sync(taxonomy, corpus):
    """Rebuild question_count from corpus primary_concept."""
    actual = Counter(q.get("primary_concept", "") for q in corpus)
    fixed = 0
    for c in taxonomy["concepts"]:
        real = actual.get(c["id"], 0)
        if c.get("question_count", 0) != real:
            c["question_count"] = real
            fixed += 1
    return fixed


def fix_09_canonical_levels(corpus):
    """Normalize L6 -> L6+."""
    fixed = 0
    for q in corpus:
        if q.get("level") == "L6":
            q["level"] = "L6+"
            fixed += 1
    return fixed


# ── Runner ───────────────────────────────────────────────────

ALL_CHECKS = [
    check_01_duplicate_concept_names,
    check_02_duplicate_concept_ids,
    check_03_kebab_case_ids,
    check_04_question_count_sync,
    check_05_corpus_concepts_in_taxonomy,
    check_06_prerequisite_integrity,
    check_07_no_cycles,
    check_08_canonical_competency_area,
    check_09_canonical_levels,
    check_10_duplicate_question_ids,
    check_11_chain_question_ids,
    check_12_title_uniqueness,
    check_13_singleton_concepts,
    check_14_corpus_only_concepts,
    check_15_zone_coverage,
    check_16_topic_coverage,
    check_17_topic_concentration,
    check_18_chain_levels_populated,
    check_19_validated_consistency,
]

FIXERS = {
    4: fix_04_question_count_sync,
    9: fix_09_canonical_levels,
}


def main():
    parser = argparse.ArgumentParser(description="Vault invariant checks")
    parser.add_argument("--fix", action="store_true", help="Auto-fix fixable issues")
    parser.add_argument("--check", type=int, help="Run only this check number")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    corpus, taxonomy, chains = load_data()

    checks_to_run = ALL_CHECKS
    if args.check:
        checks_to_run = [c for c in ALL_CHECKS if c.__name__.startswith(f"check_{args.check:02d}")]
        if not checks_to_run:
            print(f"No check #{args.check} found")
            sys.exit(1)

    results = []
    for check_fn in checks_to_run:
        result = check_fn(taxonomy=taxonomy, corpus=corpus, chains=chains)
        results.append(result)

    # Auto-fix pass
    if args.fix:
        modified_corpus = False
        modified_taxonomy = False

        for result in results:
            if result.status == "FAIL" and result.num in FIXERS:
                fixer = FIXERS[result.num]
                # Determine which data the fixer needs
                if result.num == 4:
                    fixed = fixer(taxonomy, corpus)
                    modified_taxonomy = True
                elif result.num == 9:
                    fixed = fixer(corpus)
                    modified_corpus = True
                else:
                    fixed = 0

                if fixed:
                    result.status = "FIXED"
                    result.message += f" → auto-fixed {fixed}"

        if modified_corpus:
            with open(CORPUS_PATH, "w") as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
                f.write("\n")

        if modified_taxonomy:
            with open(TAXONOMY_PATH, "w") as f:
                json.dump(taxonomy, f, indent=2, ensure_ascii=False)
                f.write("\n")

    # Output
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        fails = 0
        warns = 0
        for r in results:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "FIXED": "🔧"}[r.status]
            color = {"PASS": "", "FAIL": "\033[91m", "WARN": "\033[93m",
                     "FIXED": "\033[92m"}[r.status]
            reset = "\033[0m" if color else ""
            fixable = " [fixable]" if r.fixable and r.status == "FAIL" else ""
            print(f"  {color}{icon} Check {r.num:>2}: {r.message}{fixable}{reset}")
            if r.status in ("FAIL", "WARN") and r.details:
                for d in r.details[:5]:
                    print(f"           {d}")
                if len(r.details) > 5:
                    print(f"           ... and {len(r.details) - 5} more")
            if r.status == "FAIL":
                fails += 1
            elif r.status == "WARN":
                warns += 1

        print()
        if fails:
            print(f"  {fails} FAIL, {warns} WARN — run with --fix to auto-fix where possible")
            sys.exit(1)
        elif warns:
            print(f"  All checks pass ({warns} warnings)")
        else:
            print("  All checks pass")

    sys.exit(1 if any(r.status == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
