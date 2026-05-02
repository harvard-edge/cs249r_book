#!/usr/bin/env python3
"""Diagnose chain coverage by (track, topic) bucket.

For each (track, topic) bucket of published questions, reports how many
questions live in the bucket and how many chains currently cover any of
them. Surfaces two lists worth a second-pass Gemini sweep:

  - ``uncovered_buckets``: ≥3 published questions, 0 chains
  - ``under_covered_buckets``: ≥6 published questions, exactly 1 chain

Output:
  - JSON sidecar at ``interviews/vault/chain-coverage.json`` (regeneratable;
    gitignored) — feeds Phase 1.4 (--buckets-from)
  - Human-readable summary on stdout: per-track totals, biggest gaps

Usage:
    python3 diagnose_chain_coverage.py
    python3 diagnose_chain_coverage.py --output path/to/coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from vault_cli.policy import is_published, load_policy

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"
POLICY_PATH = VAULT_DIR / "release-policy.yaml"
DEFAULT_OUTPUT = VAULT_DIR / "chain-coverage.json"

UNCOVERED_MIN_QUESTIONS = 3
UNDER_COVERED_MIN_QUESTIONS = 6
UNDER_COVERED_MAX_CHAINS = 1


def load_published_corpus() -> dict[str, dict]:
    policy = load_policy(POLICY_PATH)
    corpus: dict[str, dict] = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        with path.open(encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if not isinstance(d, dict) or "id" not in d:
            continue
        if not is_published(d, policy):
            continue
        corpus[d["id"]] = d
    return corpus


def bucket_corpus(corpus: dict[str, dict]) -> dict[tuple[str, str], list[str]]:
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, d in corpus.items():
        track = d.get("track")
        topic = d.get("topic")
        if not track or not topic:
            continue
        by_bucket[(track, topic)].append(qid)
    for k in by_bucket:
        by_bucket[k].sort()
    return dict(by_bucket)


def load_chains() -> list[dict]:
    with CHAINS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def chains_per_bucket(chains: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Map (track, topic) -> list of chain_ids that target that bucket."""
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in chains:
        track = c.get("track")
        topic = c.get("topic")
        cid = c.get("chain_id")
        if not (track and topic and cid):
            continue
        by_bucket[(track, topic)].append(cid)
    for k in by_bucket:
        by_bucket[k].sort()
    return dict(by_bucket)


def build_report(
    buckets: dict[tuple[str, str], list[str]],
    chains_by_bucket: dict[tuple[str, str], list[str]],
) -> dict:
    bucket_rows: list[dict] = []
    for (track, topic), qids in sorted(buckets.items()):
        cids = chains_by_bucket.get((track, topic), [])
        bucket_rows.append({
            "track": track,
            "topic": topic,
            "question_count": len(qids),
            "chain_count": len(cids),
            "qids": qids,
            "chain_ids": cids,
        })

    uncovered = [
        b for b in bucket_rows
        if b["question_count"] >= UNCOVERED_MIN_QUESTIONS and b["chain_count"] == 0
    ]
    under_covered = [
        b for b in bucket_rows
        if b["question_count"] >= UNDER_COVERED_MIN_QUESTIONS
        and b["chain_count"] <= UNDER_COVERED_MAX_CHAINS
        and b["chain_count"] > 0
    ]

    # Stable, useful ordering: most-questions-first within each list.
    uncovered.sort(key=lambda b: (-b["question_count"], b["track"], b["topic"]))
    under_covered.sort(key=lambda b: (-b["question_count"], b["track"], b["topic"]))

    return {
        "thresholds": {
            "uncovered_min_questions": UNCOVERED_MIN_QUESTIONS,
            "under_covered_min_questions": UNDER_COVERED_MIN_QUESTIONS,
            "under_covered_max_chains": UNDER_COVERED_MAX_CHAINS,
        },
        "totals": {
            "buckets": len(bucket_rows),
            "questions": sum(b["question_count"] for b in bucket_rows),
            "chains": sum(b["chain_count"] for b in bucket_rows),
            "uncovered_buckets": len(uncovered),
            "under_covered_buckets": len(under_covered),
        },
        "all_buckets": bucket_rows,
        "uncovered_buckets": uncovered,
        "under_covered_buckets": under_covered,
    }


def print_summary(report: dict) -> None:
    totals = report["totals"]
    print(f"Buckets:              {totals['buckets']}")
    print(f"Published questions:  {totals['questions']}")
    print(f"Total chains:         {totals['chains']}")
    print(f"Uncovered buckets:    {totals['uncovered_buckets']}  "
          f"(≥{UNCOVERED_MIN_QUESTIONS} questions, 0 chains)")
    print(f"Under-covered:        {totals['under_covered_buckets']}  "
          f"(≥{UNDER_COVERED_MIN_QUESTIONS} questions, ≤{UNDER_COVERED_MAX_CHAINS} chain)")

    # Per-track breakdown
    per_track: dict[str, dict[str, int]] = defaultdict(lambda: {
        "buckets": 0, "questions": 0, "chains": 0,
        "uncovered": 0, "under_covered": 0,
    })
    for b in report["all_buckets"]:
        t = per_track[b["track"]]
        t["buckets"] += 1
        t["questions"] += b["question_count"]
        t["chains"] += b["chain_count"]
    for b in report["uncovered_buckets"]:
        per_track[b["track"]]["uncovered"] += 1
    for b in report["under_covered_buckets"]:
        per_track[b["track"]]["under_covered"] += 1

    print()
    print(f"{'track':<10} {'buckets':>8} {'questions':>10} {'chains':>7} "
          f"{'chains/topic':>13} {'uncov':>6} {'undercov':>9}")
    for track in sorted(per_track):
        t = per_track[track]
        density = t["chains"] / t["buckets"] if t["buckets"] else 0.0
        print(f"{track:<10} {t['buckets']:>8} {t['questions']:>10} {t['chains']:>7} "
              f"{density:>13.2f} {t['uncovered']:>6} {t['under_covered']:>9}")

    print()
    print("Top 10 uncovered buckets by question count:")
    for b in report["uncovered_buckets"][:10]:
        print(f"  {b['track']:<8} {b['topic']:<40} q={b['question_count']}")

    if report["under_covered_buckets"]:
        print()
        print("Top 10 under-covered buckets:")
        for b in report["under_covered_buckets"][:10]:
            print(f"  {b['track']:<8} {b['topic']:<40} "
                  f"q={b['question_count']} chains={b['chain_count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    corpus = load_published_corpus()
    buckets = bucket_corpus(corpus)
    chains = load_chains()
    cbb = chains_per_bucket(chains)

    report = build_report(buckets, cbb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")

    print_summary(report)
    print()
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
