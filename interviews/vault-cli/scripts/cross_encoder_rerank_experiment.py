#!/usr/bin/env python3
"""Quick experiment: does a cross-encoder rerank improve over bi-encoder?

Uses the same calibration set (existing chains) and measures whether reranking
the top-10 bi-encoder candidates with bge-reranker-base improves precision@1.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

VAULT_DIR = Path(__file__).resolve().parents[2] / "vault"


def load_corpus():
    corpus = {}
    for path in (VAULT_DIR / "questions").rglob("*.yaml"):
        try:
            with open(path) as f: d = yaml.safe_load(f)
            if d.get("status") not in ("published", None): continue
            corpus[d["id"]] = d
        except: pass
    return corpus


def build_chains(corpus):
    chains = defaultdict(list)
    for qid, d in corpus.items():
        for c in (d.get("chains") or []):
            chains[c["id"]].append(qid)
    return chains


def emb_text(d):
    return "\n".join([
        str(d.get("scenario") or ""),
        str(d.get("question") or ""),
        str((d.get("details") or {}).get("realistic_solution") or ""),
    ]).strip()


def main():
    print("loading corpus...")
    corpus = load_corpus()
    chains = build_chains(corpus)
    print(f"  {len(corpus)} questions, {len(chains)} chains")

    # Use cached bge-small embeddings
    cache = Path(__file__).resolve().parent / ".calibration_cache" / "BAAI__bge-small-en-v1.5.npz"
    if not cache.exists():
        print(f"missing {cache} — run calibrate_chain_embeddings.py first")
        return 1
    data = np.load(cache, allow_pickle=False)
    bi_store = {str(qids[i] if hasattr(qids := data["qids"], '__getitem__') else qids[i]):
                data["vectors"][i] for i in range(len(data["qids"]))}
    bi_store = {str(data["qids"][i]): data["vectors"][i] for i in range(len(data["qids"]))}
    print(f"  bge-small embeddings: {len(bi_store)}")

    # By-bucket index
    by_bucket = defaultdict(list)
    for qid, d in corpus.items():
        by_bucket[(d.get("track"), d.get("topic"))].append(qid)

    qid_chain_set = defaultdict(set)
    for cid, members in chains.items():
        for m in members:
            qid_chain_set[m].add(cid)

    # Build evaluation set
    print("building eval set...")
    rng = random.Random(42)
    eval_items = []  # (held_qid, true_siblings, bucket_qids)
    for cid, members in chains.items():
        if len(members) < 2: continue
        held = rng.choice(members)
        if held not in bi_store: continue
        d = corpus[held]
        bucket = [q for q in by_bucket[(d.get("track"), d.get("topic"))]
                  if q != held and q in bi_store]
        if not bucket: continue
        true_siblings = set()
        for shared in qid_chain_set[held]:
            for m in chains[shared]:
                if m != held: true_siblings.add(m)
        eval_items.append((held, true_siblings, bucket))
    print(f"  eval items: {len(eval_items)}")

    # Bi-encoder baseline + top-10 candidates
    print("bi-encoder ranking...")
    p_at_1_bi = 0
    r_at_3_bi = 0
    candidates_for_rerank = []  # (held, true_sibs, top10)
    for held, true_sibs, bucket in eval_items:
        h_vec = bi_store[held]
        sims = np.array([float(h_vec @ bi_store[q]) for q in bucket])
        order = np.argsort(-sims)
        ranked = [bucket[i] for i in order]
        if ranked[0] in true_sibs: p_at_1_bi += 1
        if any(r in true_sibs for r in ranked[:3]): r_at_3_bi += 1
        candidates_for_rerank.append((held, true_sibs, ranked[:10]))
    print(f"  bge-small P@1: {p_at_1_bi/len(eval_items):.3f}")
    print(f"  bge-small R@3: {r_at_3_bi/len(eval_items):.3f}")

    # Cross-encoder rerank
    print("\nloading cross-encoder...")
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    ce = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("reranking top-10 candidates...")
    p_at_1_ce = 0
    r_at_3_ce = 0
    t0 = time.time()
    for held, true_sibs, top10 in candidates_for_rerank:
        held_text = emb_text(corpus[held])
        pairs = [(held_text, emb_text(corpus[c])) for c in top10]
        scores = ce.predict(pairs, batch_size=16, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        reranked = [top10[i] for i in order]
        if reranked[0] in true_sibs: p_at_1_ce += 1
        if any(r in true_sibs for r in reranked[:3]): r_at_3_ce += 1
    print(f"  rerank time: {time.time()-t0:.1f}s ({len(candidates_for_rerank)} items)")
    print(f"  cross-encoder P@1: {p_at_1_ce/len(eval_items):.3f}")
    print(f"  cross-encoder R@3: {r_at_3_ce/len(eval_items):.3f}")

    # Verdict
    p_gain = (p_at_1_ce - p_at_1_bi) / len(eval_items)
    r_gain = (r_at_3_ce - r_at_3_bi) / len(eval_items)
    print()
    print("=" * 60)
    print(f"P@1 gain from rerank: {p_gain:+.3f}  ({p_at_1_bi/len(eval_items):.3f} -> {p_at_1_ce/len(eval_items):.3f})")
    print(f"R@3 gain from rerank: {r_gain:+.3f}  ({r_at_3_bi/len(eval_items):.3f} -> {r_at_3_ce/len(eval_items):.3f})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
