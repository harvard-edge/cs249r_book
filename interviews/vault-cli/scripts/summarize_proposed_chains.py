#!/usr/bin/env python3
"""Summarize a proposed chains.json — distribution, stats, sample inspection.

Run after build_chains_with_gemini.py to see what was produced before
applying. Produces a quick-read text report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VAULT_DIR = Path(__file__).resolve().parents[2] / "vault"
# AI-pipeline staging lives under _pipeline/ (gitignored).
# See interviews/CLAUDE.md.
PIPELINE_DIR = VAULT_DIR / "_pipeline"
DEFAULT = PIPELINE_DIR / "chains.proposed.json"
LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT))
    ap.add_argument("--samples", type=int, default=5, help="Show N sample chains")
    args = ap.parse_args()

    chains = json.loads(Path(args.input).read_text())
    n = len(chains)
    print("=" * 60)
    print(f"PROPOSED CHAINS — {n} total")
    print("=" * 60)

    sizes = Counter(len(ch["questions"]) for ch in chains)
    print("\nchain size distribution:")
    for size in sorted(sizes):
        print(f"  size {size}: {sizes[size]} chains")

    track_counts = Counter(ch["track"] for ch in chains)
    print("\nchains per track:")
    for t, c in track_counts.most_common():
        print(f"  {t}: {c}")

    # Level deltas
    deltas = Counter()
    starts = Counter()
    for ch in chains:
        levels = [LEVEL_RANK.get(q.get("level"), 0) for q in ch["questions"]]
        starts[ch["questions"][0].get("level")] += 1
        for i in range(len(levels) - 1):
            deltas[levels[i+1] - levels[i]] += 1

    print("\nstart-level distribution:")
    for lvl in ("L1", "L2", "L3", "L4", "L5", "L6+"):
        if lvl in starts:
            print(f"  {lvl}: {starts[lvl]}")

    print("\nconsecutive-member level Δ:")
    for d, c in sorted(deltas.items()):
        bar = "█" * min(c // 10, 60)
        print(f"  Δ={d:+d}  {c:>4}  {bar}")

    # Multi-membership
    qid_count = Counter()
    for ch in chains:
        for q in ch["questions"]:
            qid_count[q["id"]] += 1
    multi = Counter(qid_count.values())
    total_chained_qids = len(qid_count)
    print("\nmulti-chain membership:")
    print(f"  total questions in any chain: {total_chained_qids}")
    for n_chains, count in sorted(multi.items()):
        if n_chains == 1:
            continue
        print(f"  in {n_chains} chains: {count} questions")

    # Topic coverage
    topics = Counter(ch["topic"] for ch in chains)
    print("\ntopic coverage:")
    print(f"  topics with at least 1 chain: {len(topics)}")
    print(f"  most-chained topic: {topics.most_common(1)[0]}")

    # Sample chains
    if args.samples and chains:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE CHAINS (first {args.samples})")
        print("=" * 60)
        for ch in chains[:args.samples]:
            levels_str = " → ".join(q["level"] for q in ch["questions"])
            print(f"\n{ch['chain_id']} | {ch['track']} | {ch['topic']}")
            print(f"  levels: {levels_str}")
            for i, q in enumerate(ch["questions"]):
                print(f"  pos {i}  {q['level']}  {q['id']}  '{q['title'][:60]}'")
            if ch.get("rationale"):
                print(f"  rationale: {ch['rationale']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
