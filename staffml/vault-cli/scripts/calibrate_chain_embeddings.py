#!/usr/bin/env python3
"""Calibrate which embedding model best discriminates real chain members.

Uses the existing 726 healthy chains as labeled ground truth:

  positives = pairs of questions in the same chain
  negatives = pairs of questions in the same (track, topic) bucket but
              different chains, OR one chained + one unchained

For each candidate embedding model, compute:
  - cosine similarity distribution on positives vs negatives
  - precision@1: hold out one member; can the model find another from the same chain?
  - recall@3:    is at least one true sibling in top-3 by similarity?
  - threshold ROC: at what cosine cutoff does precision drop below 90%?

Cache embeddings per-model so re-runs are cheap. Results written to a
JSON report; comparison table printed at the end.

Usage:
    python3 calibrate_chain_embeddings.py
    python3 calibrate_chain_embeddings.py --models bge-small,bge-base
    python3 calibrate_chain_embeddings.py --output /tmp/calib.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

VAULT_DIR = Path(__file__).resolve().parents[2] / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"
CACHE_DIR = Path(__file__).resolve().parent / ".calibration_cache"


MODELS = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base":  "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


def load_corpus() -> dict[str, dict]:
    """Load all published question YAMLs into a dict of qid -> doc."""
    corpus = {}
    for path in QUESTIONS_DIR.rglob("*.yaml"):
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
            if d.get("status") not in ("published", None):
                continue
            corpus[d["id"]] = d
        except Exception:
            continue
    return corpus


def build_chain_membership(corpus: dict[str, dict]) -> dict[str, list[str]]:
    """chain_id -> [qids] (only published members, drawn from yaml chains: field)."""
    chains = defaultdict(list)
    for qid, d in corpus.items():
        for c in (d.get("chains") or []):
            chains[c["id"]].append(qid)
    return chains


def build_pair_set(
    corpus: dict[str, dict],
    chains: dict[str, list[str]],
    sample_seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Sample positive (same-chain) and negative (same-bucket different-chain) pairs."""
    rng = random.Random(sample_seed)

    # Positives: every same-chain pair within chains of size ≥ 2
    positives = []
    for _cid, members in chains.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                positives.append((members[i], members[j]))

    # Negatives: pick from same (track, topic) but different chain
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    qid_to_chains: dict[str, set[str]] = defaultdict(set)
    for qid, d in corpus.items():
        by_bucket[(d.get("track"), d.get("topic"))].append(qid)
        for c in (d.get("chains") or []):
            qid_to_chains[qid].add(c["id"])

    negatives = []
    target_neg = len(positives)  # match positive count for balance
    attempts = 0
    while len(negatives) < target_neg and attempts < target_neg * 20:
        attempts += 1
        bucket_qids = rng.choice(list(by_bucket.values()))
        if len(bucket_qids) < 2:
            continue
        a, b = rng.sample(bucket_qids, 2)
        # Negative if: their chain sets are disjoint
        if qid_to_chains[a] & qid_to_chains[b]:
            continue
        negatives.append((a, b))

    return positives, negatives


def embed_corpus(
    corpus: dict[str, dict],
    model_name: str,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], float]:
    """Return (qid -> vector) and seconds-elapsed. Caches per-model to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "__")
    cache_path = cache_dir / f"{safe}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        qids = data["qids"]
        vecs = data["vectors"]
        store = {str(qids[i]): vecs[i] for i in range(len(qids))}
        # Re-run if corpus has any qid we don't have cached
        missing = set(corpus) - set(store)
        if not missing:
            print(f"  [{model_name}] cached ({len(store)} embeddings)")
            return store, 0.0
        print(f"  [{model_name}] cache stale, missing {len(missing)} — re-embedding")

    print(f"  [{model_name}] embedding {len(corpus)} questions...")
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(model_name)
    qids = sorted(corpus.keys())
    texts = []
    for qid in qids:
        d = corpus[qid]
        text = "\n".join([
            str(d.get("scenario") or ""),
            str(d.get("question") or ""),
            str((d.get("details") or {}).get("realistic_solution") or ""),
        ]).strip()
        texts.append(text)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    elapsed = time.time() - t0

    qids_arr = np.array(qids)
    vecs_arr = np.asarray(vecs, dtype=np.float32)
    np.savez(cache_path, qids=qids_arr, vectors=vecs_arr)
    store = {qids[i]: vecs_arr[i] for i in range(len(qids))}
    print(f"  [{model_name}] {elapsed:.1f}s, dim={vecs_arr.shape[1]}, cached -> {cache_path.name}")
    return store, elapsed


def evaluate(
    store: dict[str, np.ndarray],
    corpus: dict[str, dict],
    chains: dict[str, list[str]],
    positives: list[tuple[str, str]],
    negatives: list[tuple[str, str]],
) -> dict:
    """Run the metric suite for one model's embeddings."""
    # 1. Cosine distributions
    pos_sims = np.array([float(store[a] @ store[b]) for a, b in positives if a in store and b in store])
    neg_sims = np.array([float(store[a] @ store[b]) for a, b in negatives if a in store and b in store])

    # 2. Precision@1 and Recall@3 via leave-one-out per chain
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, d in corpus.items():
        by_bucket[(d.get("track"), d.get("topic"))].append(qid)

    qid_chains: dict[str, set[str]] = defaultdict(set)
    for cid, members in chains.items():
        for m in members:
            qid_chains[m].add(cid)

    p_at_1_hits = p_at_1_total = 0
    r_at_3_hits = r_at_3_total = 0

    for _cid, members in chains.items():
        if len(members) < 2:
            continue
        for held in members:
            if held not in store:
                continue
            held_doc = corpus[held]
            bucket = by_bucket[(held_doc.get("track"), held_doc.get("topic"))]
            other_in_bucket = [q for q in bucket if q != held and q in store]
            if not other_in_bucket:
                continue
            held_vec = store[held]
            sims = np.array([float(held_vec @ store[q]) for q in other_in_bucket])
            order = np.argsort(-sims)
            ranked = [other_in_bucket[i] for i in order]

            # ground truth: any other member of any chain that contains `held`
            true_siblings = set()
            for shared_chain in qid_chains[held]:
                for m in chains[shared_chain]:
                    if m != held:
                        true_siblings.add(m)
            if not true_siblings:
                continue

            p_at_1_total += 1
            r_at_3_total += 1
            if ranked[0] in true_siblings:
                p_at_1_hits += 1
            if any(r in true_siblings for r in ranked[:3]):
                r_at_3_hits += 1

    # 3. Threshold ROC: at each cosine cutoff τ, what fraction of pairs ≥ τ are positives?
    all_sims = list(pos_sims) + list(neg_sims)
    all_labels = [1] * len(pos_sims) + [0] * len(neg_sims)
    arr_sims = np.array(all_sims)
    arr_labels = np.array(all_labels)
    thresholds = np.linspace(0.5, 0.95, 19)
    threshold_table = []
    for t in thresholds:
        mask = arr_sims >= t
        n = int(mask.sum())
        precision = float(arr_labels[mask].sum() / n) if n else 0.0
        recall = float(arr_labels[mask].sum() / arr_labels.sum()) if arr_labels.sum() else 0.0
        threshold_table.append({"tau": float(t), "n": n, "precision": precision, "recall": recall})

    # Find τ where precision crosses 90%
    tau_p90 = None
    for row in threshold_table:
        if row["precision"] >= 0.90:
            tau_p90 = row["tau"]
            break

    return {
        "n_positive_pairs": int(len(pos_sims)),
        "n_negative_pairs": int(len(neg_sims)),
        "pos_cosine_mean": float(pos_sims.mean()) if len(pos_sims) else 0.0,
        "pos_cosine_std":  float(pos_sims.std())  if len(pos_sims) else 0.0,
        "neg_cosine_mean": float(neg_sims.mean()) if len(neg_sims) else 0.0,
        "neg_cosine_std":  float(neg_sims.std())  if len(neg_sims) else 0.0,
        "separation":      float(pos_sims.mean() - neg_sims.mean()) if len(pos_sims) and len(neg_sims) else 0.0,
        "precision_at_1":  p_at_1_hits / p_at_1_total if p_at_1_total else 0.0,
        "recall_at_3":     r_at_3_hits / r_at_3_total if r_at_3_total else 0.0,
        "tau_for_precision_90": tau_p90,
        "threshold_table": threshold_table,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="bge-small,bge-base,bge-large,minilm-l6",
                    help="Comma-separated keys from MODELS dict")
    ap.add_argument("--output", default=None, help="Write JSON report to this path")
    args = ap.parse_args()

    print("loading corpus + chains...")
    corpus = load_corpus()
    chains = build_chain_membership(corpus)
    print(f"  corpus: {len(corpus)} published questions")
    print(f"  chains: {len(chains)} (sizes: min={min(len(m) for m in chains.values()) if chains else 0}, "
          f"max={max(len(m) for m in chains.values()) if chains else 0})")

    print("\nbuilding pair set...")
    positives, negatives = build_pair_set(corpus, chains)
    print(f"  positives (same-chain): {len(positives)}")
    print(f"  negatives (same-bucket different-chain): {len(negatives)}")

    requested = args.models.split(",")
    results = {}
    for key in requested:
        key = key.strip()
        if key not in MODELS:
            print(f"unknown model: {key}; skipping", file=sys.stderr)
            continue
        model_name = MODELS[key]
        print(f"\n=== {key} ({model_name}) ===")
        store, elapsed = embed_corpus(corpus, model_name, CACHE_DIR)
        metrics = evaluate(store, corpus, chains, positives, negatives)
        metrics["embedding_seconds"] = elapsed
        metrics["model_key"] = key
        metrics["model_name"] = model_name
        results[key] = metrics

        print(f"  pos cosine: μ={metrics['pos_cosine_mean']:.3f} σ={metrics['pos_cosine_std']:.3f}")
        print(f"  neg cosine: μ={metrics['neg_cosine_mean']:.3f} σ={metrics['neg_cosine_std']:.3f}")
        print(f"  separation: {metrics['separation']:+.3f}")
        print(f"  precision@1: {metrics['precision_at_1']:.3f}")
        print(f"  recall@3:    {metrics['recall_at_3']:.3f}")
        print(f"  τ for precision≥0.9: {metrics['tau_for_precision_90']}")

    print("\n" + "=" * 88)
    print(f"{'model':<14} {'sep':>7} {'P@1':>7} {'R@3':>7} {'τ p=0.9':>9} {'embed sec':>11}")
    print("-" * 88)
    for key, m in results.items():
        tau = f"{m['tau_for_precision_90']:.2f}" if m['tau_for_precision_90'] else " n/a"
        print(f"{key:<14} {m['separation']:>+7.3f} {m['precision_at_1']:>7.3f} "
              f"{m['recall_at_3']:>7.3f} {tau:>9} {m['embedding_seconds']:>11.1f}")
    print("=" * 88)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nfull report -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
