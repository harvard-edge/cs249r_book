"""Rescue suggestions for orphan singleton chains.

For each orphan, find ranked candidate questions in the same (track, topic)
bucket that could plausibly extend the chain. Honors structural constraints:
single-topic, Bloom-monotonic level adjacency, candidate not already chained.

Pure embedding-based ranking — no LLM. Confidence bands derived from
Phase 0 calibration on existing chains:

  strong-merge:   cosine >= τ_strong AND |level_delta| <= 1
  review-merge:   cosine in [τ_review, τ_strong] OR |level_delta| == 2
  blocked:        structurally invalid (filters reject before ranking)

Empirical note from calibration on the existing 726 chains:
  - bge-small precision@1 ≈ 0.28
  - bge-small recall@3    ≈ 0.45
Same-bucket questions are inherently close in embedding space; therefore
this command outputs RANKED CANDIDATES for human review, never auto-applies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vault_cli.chains.audit import build_chain_view, load_corpus
from vault_cli.chains.embeddings import EmbeddingStore

LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}


@dataclass
class RescueCandidate:
    candidate_id: str
    cosine: float
    level: str
    level_delta: int
    band: str  # strong-merge | review-merge | blocked


@dataclass
class OrphanRescue:
    orphan_id: str
    chain_id: str
    track: str
    topic: str
    level: str
    candidates: list[RescueCandidate]


def suggest_rescues(
    vault_dir: Path,
    store: EmbeddingStore,
    *,
    tau_strong: float = 0.85,
    tau_review: float = 0.70,
    top_k: int = 5,
) -> list[OrphanRescue]:
    """Compute ranked rescue candidates for each orphan singleton chain.

    Hybrid approach: hard structural filter on level adjacency, then soft
    ranking by embedding cosine within the structurally-valid candidate set.

    Level filter is direction-aware:
      Empirically, 100% of existing chains are non-decreasing in level
      and 92% have consecutive Δ ∈ {1, 2}. So forward extensions are
      orphan_level → cand_level with delta ∈ {0, 1, 2}; backward
      extensions are cand_level → orphan_level (same constraint).
      We allow both since we don't know whether orphan was first or last
      member of its dead chain. Δ=1 gets strong-band priority.
    """
    corpus = load_corpus(vault_dir)
    chains = build_chain_view(corpus)

    # Bucket: (track, topic) -> [qid]
    by_bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    chained: set[str] = set()
    for qid, d in corpus.items():
        by_bucket[(d.get("track"), d.get("topic"))].append(qid)
        if d.get("chains"):
            chained.add(qid)

    rescues: list[OrphanRescue] = []
    for cid, members in chains.items():
        if len(members) >= 2:
            continue
        orphan_qid, _ = members[0]
        orphan_doc = corpus.get(orphan_qid)
        if not orphan_doc: continue
        if store.get(orphan_qid) is None: continue

        track = orphan_doc.get("track")
        topic = orphan_doc.get("topic")
        orphan_level = orphan_doc.get("level", "?")
        orphan_rank = LEVEL_RANK.get(orphan_level, 0)
        orphan_vec = store.get(orphan_qid)

        bucket = by_bucket.get((track, topic), [])
        cands: list[RescueCandidate] = []
        for cqid in bucket:
            if cqid == orphan_qid: continue
            if cqid in chained: continue
            cdoc = corpus.get(cqid)
            if not cdoc: continue
            cvec = store.get(cqid)
            if cvec is None: continue
            cand_level = cdoc.get("level", "?")
            cand_rank = LEVEL_RANK.get(cand_level, 0)
            signed_delta = cand_rank - orphan_rank
            level_delta = abs(signed_delta)

            # Hard filter: level delta must match observed chain patterns.
            # 92% of edges are Δ ∈ {1, 2}; 3% are Δ=0; 5% are Δ≥3.
            if level_delta > 2: continue

            cosine = float(orphan_vec @ cvec)

            # Band assignment honors empirical level structure:
            # - Δ=1 + high cosine    -> strong-merge (most common chain shape)
            # - Δ=2 OR moderate cos  -> review-merge (still plausible)
            # - same level (Δ=0)     -> review only (uncommon in chains)
            if cosine >= tau_strong and level_delta == 1:
                band = "strong-merge"
            elif cosine >= tau_review and level_delta in (1, 2):
                band = "review-merge"
            elif cosine >= tau_review and level_delta == 0:
                band = "review-merge-same-level"
            else:
                band = "below-threshold"

            cands.append(RescueCandidate(
                candidate_id=cqid, cosine=cosine, level=cand_level,
                level_delta=signed_delta, band=band,
            ))

        # Sort: prioritize Δ=1, then cosine
        cands.sort(key=lambda c: (-(1 if abs(c.level_delta) == 1 else 0), -c.cosine))
        rescues.append(OrphanRescue(
            orphan_id=orphan_qid,
            chain_id=cid,
            track=track,
            topic=topic,
            level=orphan_level,
            candidates=cands[:top_k],
        ))

    return rescues


def format_rescue_report(rescues: list[OrphanRescue]) -> str:
    if not rescues:
        return "no orphan singletons — corpus is clean."
    lines = []
    lines.append("=" * 80)
    lines.append(f"ORPHAN RESCUE SUGGESTIONS — {len(rescues)} orphans")
    lines.append("=" * 80)
    band_counts = defaultdict(int)
    for r in rescues:
        for c in r.candidates:
            band_counts[c.band] += 1
    lines.append(f"  candidates by band: {dict(band_counts)}")
    lines.append("")
    for r in rescues:
        lines.append(f"orphan: {r.orphan_id} (level {r.level}, track={r.track}, topic={r.topic})")
        lines.append(f"  chain {r.chain_id} reduced to 1 member")
        if not r.candidates:
            lines.append("  no candidates within bucket")
        for c in r.candidates:
            marker = "★" if c.band == "strong-merge" else "·" if c.band == "review-merge" else " "
            lines.append(
                f"  {marker} {c.candidate_id:<14} cos={c.cosine:.3f} "
                f"level={c.level} (Δ={c.level_delta}) [{c.band}]"
            )
        lines.append("")
    lines.append("=" * 80)
    lines.append("LEGEND: ★ strong-merge (auto-suggest), · review-merge (worth checking)")
    lines.append("This is a suggestion list; authors should manually accept proposed merges.")
    return "\n".join(lines)


def rescues_to_dict(rescues: list[OrphanRescue]) -> list[dict]:
    return [asdict(r) for r in rescues]


__all__ = [
    "OrphanRescue",
    "RescueCandidate",
    "format_rescue_report",
    "rescues_to_dict",
    "suggest_rescues",
]
