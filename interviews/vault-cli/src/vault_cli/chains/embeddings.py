"""Sentence embeddings for chain similarity.

Embeds (scenario + question + realistic_solution) with BGE-small. Caches
to a content-hashed sidecar so re-runs only re-embed changed questions.

The sidecar is gitignored (large, reproducible). A small manifest is
committed (interviews/vault/embeddings-manifest.json) recording the
content-hash → vector-hash map; CI uses it to skip embedding when source
yamls are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMB_DIM = 384  # bge-small-en dim


@dataclass(frozen=True)
class EmbeddingStore:
    """Maps qid → (embedding vector, content_hash)."""

    vectors: dict[str, np.ndarray]
    content_hashes: dict[str, str]
    model_name: str

    def get(self, qid: str) -> np.ndarray | None:
        return self.vectors.get(qid)


def _content_hash(question_doc: dict) -> str:
    """Stable hash over the embedded fields (so we re-embed only on real changes)."""
    parts = [
        str(question_doc.get("scenario", "")),
        str(question_doc.get("question", "")),
        str((question_doc.get("details") or {}).get("realistic_solution", "")),
    ]
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _embedding_text(question_doc: dict) -> str:
    """Concatenated text to embed. Order matches _content_hash."""
    return "\n".join(
        [
            str(question_doc.get("scenario", "")),
            str(question_doc.get("question", "")),
            str((question_doc.get("details") or {}).get("realistic_solution", "")),
        ]
    ).strip()


def load_or_build(
    questions: list[dict],
    sidecar_path: Path,
    model_name: str = DEFAULT_MODEL,
    *,
    progress: bool = True,
) -> EmbeddingStore:
    """Load embeddings from sidecar, computing only those whose content changed.

    Sidecar format: NPZ with keys:
      - 'vectors': float32 array (N, D)
      - 'qids': string array (N,)
      - 'content_hashes': string array (N,) — what was hashed when each was computed
      - 'model_name': string scalar
    """
    # Compute desired content hashes for all current questions
    target = {q["id"]: (_content_hash(q), _embedding_text(q)) for q in questions}

    # Load any existing sidecar
    cached: dict[str, tuple[str, np.ndarray]] = {}
    cached_model = None
    if sidecar_path.exists():
        try:
            data = np.load(sidecar_path, allow_pickle=False)
            qids = data["qids"]
            hashes = data["content_hashes"]
            vecs = data["vectors"]
            cached_model = str(data["model_name"])
            for i, qid in enumerate(qids):
                cached[str(qid)] = (str(hashes[i]), vecs[i])
        except Exception:
            cached = {}

    # If model changed, invalidate cache entirely
    if cached_model != model_name:
        cached = {}

    # Decide what needs (re-)embedding
    to_embed: list[tuple[str, str]] = []
    valid_cached: dict[str, np.ndarray] = {}
    for qid, (target_hash, text) in target.items():
        if qid in cached and cached[qid][0] == target_hash:
            valid_cached[qid] = cached[qid][1]
        else:
            to_embed.append((qid, text))

    if to_embed:
        if progress:
            print(f"  embedding {len(to_embed)} of {len(target)} (using cached: {len(valid_cached)})")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        new_vecs = model.encode(
            [t for _, t in to_embed],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=progress,
        )
        for (qid, _), vec in zip(to_embed, new_vecs):
            valid_cached[qid] = np.asarray(vec, dtype=np.float32)
    elif progress:
        print(f"  all {len(target)} embeddings cached — no re-compute")

    # Save updated sidecar
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    qids_arr = np.array(list(valid_cached.keys()))
    vecs_arr = np.stack([valid_cached[q] for q in qids_arr]) if qids_arr.size else np.zeros((0, EMB_DIM), dtype=np.float32)
    hashes_arr = np.array([target[q][0] for q in qids_arr])
    np.savez(
        sidecar_path,
        vectors=vecs_arr,
        qids=qids_arr,
        content_hashes=hashes_arr,
        model_name=np.array(model_name),
    )

    return EmbeddingStore(
        vectors=valid_cached,
        content_hashes={q: target[q][0] for q in valid_cached},
        model_name=model_name,
    )


def write_manifest(store: EmbeddingStore, manifest_path: Path) -> None:
    """Lightweight committed manifest: which qids were embedded under which content hash."""
    manifest = {
        "model_name": store.model_name,
        "embedding_dim": EMB_DIM,
        "count": len(store.vectors),
        "content_hashes": dict(sorted(store.content_hashes.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


__all__ = [
    "DEFAULT_MODEL",
    "EMB_DIM",
    "EmbeddingStore",
    "load_or_build",
    "write_manifest",
]
