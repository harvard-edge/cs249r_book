"""Chain audit: orphans, position drift, stale registry, similarity drift.

Reads the YAML corpus + chains.json, computes a current snapshot of chain
health, and emits a report. Read-only — no mutations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from vault_cli.chains.embeddings import EmbeddingStore


@dataclass
class ChainHealth:
    chain_id: str
    track: str
    topic: str
    member_count: int
    members: list[str]
    levels: list[str]
    intra_chain_cosine_mean: float | None = None
    intra_chain_cosine_min: float | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    total_questions: int
    total_chains: int
    chain_size_distribution: dict[int, int]
    orphan_singletons: list[str]
    position_drift: list[str]
    stale_registry: list[str]
    chain_quality_p10: float | None
    chain_quality_p50: float | None
    chain_quality_p90: float | None
    weak_chains_top: list[ChainHealth]
    healthy_chain_count: int


def load_corpus(vault_dir: Path) -> dict[str, dict]:
    """qid -> yaml dict, published only."""
    corpus = {}
    for path in (vault_dir / "questions").rglob("*.yaml"):
        try:
            with open(path) as f:
                d = yaml.safe_load(f)
            if d.get("status") not in ("published", None):
                continue
            corpus[d["id"]] = d
        except Exception:
            continue
    return corpus


def build_chain_view(corpus: dict[str, dict]) -> dict[str, list[tuple[str, int]]]:
    """chain_id -> [(qid, position)] from yaml chains: field."""
    chains: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for qid, d in corpus.items():
        for c in (d.get("chains") or []):
            chains[c["id"]].append((qid, c.get("position", 0)))
    return chains


def load_registry(vault_dir: Path) -> set[str]:
    """chain_ids declared in chains.json."""
    path = vault_dir / "chains.json"
    if not path.exists():
        return set()
    return {ch.get("chain_id") for ch in json.loads(path.read_text())}


def run_audit(vault_dir: Path, store: EmbeddingStore | None = None) -> AuditReport:
    """Compute the audit report. Pass `store` to enable similarity-drift metrics."""
    corpus = load_corpus(vault_dir)
    chains = build_chain_view(corpus)
    registry = load_registry(vault_dir)

    sizes = defaultdict(int)
    orphans: list[str] = []
    position_drift: list[str] = []
    chain_healths: list[ChainHealth] = []

    for cid, members in chains.items():
        sizes[len(members)] += 1
        if len(members) < 2:
            orphans.append(cid)
        positions = sorted(p for _, p in members)
        if positions != list(range(len(positions))):
            position_drift.append(cid)

        # First-pass health record
        first_member_doc = corpus.get(members[0][0])
        if first_member_doc is None:
            continue
        ch = ChainHealth(
            chain_id=cid,
            track=first_member_doc.get("track"),
            topic=first_member_doc.get("topic"),
            member_count=len(members),
            members=[qid for qid, _ in sorted(members, key=lambda m: m[1])],
            levels=[corpus.get(qid, {}).get("level", "?") for qid, _ in sorted(members, key=lambda m: m[1])],
        )
        if cid in orphans:
            ch.issues.append("orphan-singleton")
        if cid in position_drift:
            ch.issues.append("position-drift")
        chain_healths.append(ch)

    # Stale registry: chains.json entries with no live members in yamls
    stale = sorted(registry - set(chains.keys()))

    # Similarity drift if embeddings available: mean & min intra-chain cosine
    quality_scores: list[float] = []
    if store is not None:
        for ch in chain_healths:
            if ch.member_count < 2:
                continue
            vecs = [store.get(m) for m in ch.members]
            if any(v is None for v in vecs):
                continue
            arr = np.stack(vecs)
            sim = arr @ arr.T  # symmetric, diag = 1
            n = len(vecs)
            triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            pairs = sim[triu_mask]
            ch.intra_chain_cosine_mean = float(pairs.mean())
            ch.intra_chain_cosine_min  = float(pairs.min())
            quality_scores.append(ch.intra_chain_cosine_mean)

    p10 = float(np.percentile(quality_scores, 10)) if quality_scores else None
    p50 = float(np.percentile(quality_scores, 50)) if quality_scores else None
    p90 = float(np.percentile(quality_scores, 90)) if quality_scores else None

    weak = sorted(
        [ch for ch in chain_healths if ch.intra_chain_cosine_mean is not None],
        key=lambda ch: ch.intra_chain_cosine_mean,
    )[:10]

    return AuditReport(
        total_questions=len(corpus),
        total_chains=len(chains),
        chain_size_distribution=dict(sorted(sizes.items())),
        orphan_singletons=sorted(orphans),
        position_drift=sorted(position_drift),
        stale_registry=stale,
        chain_quality_p10=p10,
        chain_quality_p50=p50,
        chain_quality_p90=p90,
        weak_chains_top=weak,
        healthy_chain_count=sum(1 for ch in chain_healths if not ch.issues),
    )


def format_text_report(rep: AuditReport) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("VAULT CHAIN AUDIT")
    lines.append("=" * 64)
    lines.append(f"  questions (published): {rep.total_questions}")
    lines.append(f"  chains:                {rep.total_chains}")
    lines.append(f"  healthy:               {rep.healthy_chain_count}")
    lines.append("")
    lines.append(f"  chain size distribution: {rep.chain_size_distribution}")
    lines.append("")
    lines.append(f"  orphan singletons (size<2): {len(rep.orphan_singletons)}")
    if rep.orphan_singletons:
        lines.append(f"    {', '.join(rep.orphan_singletons[:8])}"
                     + (f" (and {len(rep.orphan_singletons)-8} more)" if len(rep.orphan_singletons) > 8 else ""))
    lines.append(f"  position drift:           {len(rep.position_drift)}")
    if rep.position_drift:
        lines.append(f"    {', '.join(rep.position_drift[:8])}"
                     + (f" (and {len(rep.position_drift)-8} more)" if len(rep.position_drift) > 8 else ""))
    lines.append(f"  stale registry entries:   {len(rep.stale_registry)}")
    if rep.stale_registry:
        lines.append(f"    {', '.join(rep.stale_registry[:8])}"
                     + (f" (and {len(rep.stale_registry)-8} more)" if len(rep.stale_registry) > 8 else ""))
    lines.append("")
    if rep.chain_quality_p50 is not None:
        lines.append("  chain similarity (intra-chain cosine):")
        lines.append(f"    p10 (weakest chains):  {rep.chain_quality_p10:.3f}")
        lines.append(f"    p50 (median):           {rep.chain_quality_p50:.3f}")
        lines.append(f"    p90 (tightest chains): {rep.chain_quality_p90:.3f}")
        if rep.weak_chains_top:
            lines.append("")
            lines.append("  WEAKEST 5 chains (lowest intra-chain cosine):")
            for ch in rep.weak_chains_top[:5]:
                lines.append(f"    {ch.chain_id}: cos={ch.intra_chain_cosine_mean:.3f} "
                             f"({ch.member_count} members, topic={ch.topic})")
    lines.append("=" * 64)
    return "\n".join(lines)


def report_to_dict(rep: AuditReport) -> dict:
    d = asdict(rep)
    return d


__all__ = [
    "AuditReport",
    "ChainHealth",
    "build_chain_view",
    "format_text_report",
    "load_corpus",
    "load_registry",
    "report_to_dict",
    "run_audit",
]
