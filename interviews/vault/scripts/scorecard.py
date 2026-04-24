#!/usr/bin/env python3
"""
StaffML Quality Scorecard — single-command corpus health check.

Usage:
    python3 staffml/vault/scripts/scorecard.py
    python3 staffml/vault/scripts/scorecard.py --compare path/to/previous.json
"""

import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

import yaml


VAULT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_DIR = VAULT_DIR / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from enums import VALID_TOPICS  # noqa: E402


def load_yaml_overrides() -> dict[str, dict]:
    """Load source YAML fields that legacy corpus.json may omit."""
    out: dict[str, dict] = {}
    questions_dir = VAULT_DIR / "questions"
    if not questions_dir.exists():
        return out
    for path in questions_dir.glob("*/*.yaml"):
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        qid = data.get("id")
        if qid:
            out[qid] = data
    return out


def compute_scorecard(corpus_path: str, chains_path: str, taxonomy_path: str) -> dict:
    with open(corpus_path) as f:
        data = json.load(f)
    with open(chains_path) as f:
        chains = json.load(f)
    with open(taxonomy_path) as f:
        tax = json.load(f)

    yaml_by_id = load_yaml_overrides()
    for q in data:
        source = yaml_by_id.get(q.get("id"), {})
        for field in (
            "question",
            "validation_status",
            "validated",
            "math_verified",
            "human_reviewed",
            "provenance",
        ):
            if field not in q and field in source:
                q[field] = source[field]

    tax_ids = {c["id"] for c in tax["concepts"]}
    schema_topic_ids = set(VALID_TOPICS)
    id_to_q = {q["id"]: q for q in data}

    sc = {"timestamp": datetime.now().isoformat(), "total": len(data)}

    # Validation status
    vs = Counter(q.get("validation_status", "unknown") for q in data)
    sc["validation"] = {
        "OK": vs.get("OK", 0),
        "WARN": vs.get("WARN", 0),
        "ERROR": vs.get("ERROR", 0),
        "pending": sum(vs.get(s, 0) for s in ("pending", "unknown", "", None)),
        "ok_pct": round(100 * vs.get("OK", 0) / len(data), 1),
    }

    # Per-track counts
    tracks = ["cloud", "edge", "mobile", "tinyml", "global"]
    sc["tracks"] = {}
    for t in tracks:
        qs = [q for q in data if q["track"] == t]
        chained = sum(1 for q in qs if q.get("chain_ids"))
        sc["tracks"][t] = {
            "count": len(qs),
            "chain_coverage_pct": round(100 * chained / len(qs), 1) if qs else 0,
        }

    # Bloom balance
    ideal = {"remember": 0.10, "understand": 0.15, "apply": 0.25, "analyze": 0.25, "evaluate": 0.15, "create": 0.10}
    sc["bloom_deviation"] = {}
    for t in tracks:
        qs = [q for q in data if q["track"] == t]
        n = len(qs)
        if n == 0:
            continue
        blooms = Counter(q.get("bloom_level", "?") for q in qs)
        dev = sum(abs(blooms.get(b, 0) / n - p) for b, p in ideal.items()) / len(ideal)
        sc["bloom_deviation"][t] = round(dev * 100, 1)

    # Taxonomy (v1): `topic` is the canonical per-question concept field.
    empty_topic = sum(1 for q in data if not q.get("topic", "").strip())
    corpus_topics = {q.get("topic", "") for q in data if q.get("topic", "").strip()}
    missing_topic = sorted(corpus_topics - schema_topic_ids)
    zero_schema_topics = sorted(t for t in schema_topic_ids if t not in corpus_topics)
    sc["taxonomy"] = {
        "taxonomy_json_concepts": len(tax["concepts"]),
        "schema_topics": len(schema_topic_ids),
        "edges": len(tax.get("edges", [])),
        "empty_topic_refs": empty_topic,
        "corpus_topics_not_in_schema": missing_topic,
        "schema_topics_with_zero_questions": zero_schema_topics,
        "taxonomy_json_overlap": len(corpus_topics & tax_ids),
    }

    # Quality
    short_sol = sum(1 for q in data if len(q.get("details", {}).get("realistic_solution", "") or "") < 100)
    short_nm = sum(1 for q in data if len(q.get("details", {}).get("napkin_math", "") or "") < 50)
    full_v1_axes = sum(
        1
        for q in data
        if all(
            [
                q.get("track"),
                q.get("level"),
                q.get("zone"),
                q.get("topic"),
                q.get("competency_area"),
                q.get("bloom_level"),
                q.get("phase"),
            ]
        )
    )
    question_field = sum(1 for q in data if (q.get("question") or "").strip())
    sc["quality"] = {
        "short_solutions": short_sol,
        "short_napkin_math": short_nm,
        "full_v1_axes": full_v1_axes,
        "full_v1_axes_pct": round(100 * full_v1_axes / len(data), 1),
        "question_field_populated": question_field,
        "question_field_pct": round(100 * question_field / len(data), 1),
    }

    # Chains
    sc["chains"] = {
        "total": len(chains),
        "coverage": sum(1 for q in data if q.get("chain_ids")),
        "coverage_pct": round(100 * sum(1 for q in data if q.get("chain_ids")) / len(data), 1),
    }

    return sc


def print_scorecard(sc: dict, prev: dict = None):
    def delta(key_path, sc, prev):
        keys = key_path.split(".")
        v = sc
        p = prev
        for k in keys:
            v = v.get(k, {}) if isinstance(v, dict) else None
            p = p.get(k, {}) if isinstance(p, dict) else None
        if v is not None and p is not None and isinstance(v, (int, float)) and isinstance(p, (int, float)):
            d = v - p
            return f" ({'+' if d >= 0 else ''}{d})" if d != 0 else ""
        return ""

    print("=" * 60)
    print(f"STAFFML SCORECARD — {sc['timestamp'][:10]}")
    print("=" * 60)
    print(f"\nQuestions: {sc['total']}")

    v = sc["validation"]
    print(f"\nValidation:")
    print(f"  OK:    {v['OK']:>5} ({v['ok_pct']}%){delta('validation.OK', sc, prev or {})}")
    print(f"  WARN:  {v['WARN']:>5}{delta('validation.WARN', sc, prev or {})}")
    print(f"  ERROR: {v['ERROR']:>5}")
    print(f"  Pend:  {v['pending']:>5}")

    print(f"\nChains: {sc['chains']['total']} | Coverage: {sc['chains']['coverage_pct']}%")
    for t, d in sc["tracks"].items():
        print(f"  {t}: {d['count']} Qs, {d['chain_coverage_pct']}% chained")

    print(f"\nBloom deviation (lower=better, target <3%):")
    for t, d in sc["bloom_deviation"].items():
        print(f"  {t}: {d}%")

    print(
        f"\nTaxonomy: {sc['taxonomy']['schema_topics']} schema topics, "
        f"{sc['taxonomy']['taxonomy_json_concepts']} taxonomy.json concepts"
    )
    print(f"  Empty topic refs: {sc['taxonomy']['empty_topic_refs']}")
    print(f"  Corpus topics not in schema: {len(sc['taxonomy']['corpus_topics_not_in_schema'])}")
    print(f"  Schema topics with zero questions: {len(sc['taxonomy']['schema_topics_with_zero_questions'])}")

    print(f"\nQuality:")
    print(f"  v1 axes complete: {sc['quality']['full_v1_axes_pct']}%")
    print(f"  Question field populated: {sc['quality']['question_field_pct']}%")
    print(f"  Short solutions: {sc['quality']['short_solutions']}")
    print(f"  Short napkin: {sc['quality']['short_napkin_math']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus.json")
    parser.add_argument("--chains", default="chains.json")
    parser.add_argument("--taxonomy", default="taxonomy.json")
    parser.add_argument("--compare", help="Path to previous scorecard JSON for delta")
    parser.add_argument("--output", help="Save scorecard JSON to this path")
    args = parser.parse_args()

    sc = compute_scorecard(args.corpus, args.chains, args.taxonomy)

    prev = None
    if args.compare:
        with open(args.compare) as f:
            prev = json.load(f)

    print_scorecard(sc, prev)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(sc, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        # Default output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"_validation_results/scorecard_{ts}.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(sc, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
