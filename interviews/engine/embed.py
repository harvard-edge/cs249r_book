"""
Vector embedding and gap analysis engine.

Uses ChromaDB + sentence-transformers to:
1. Embed all existing questions into a vector store
2. Embed textbook chapter concepts
3. Identify semantic gaps (concepts with low question coverage)
4. Prevent duplicate question generation

Embedding model: nomic-embed-text-v1.5 (137M params, 8192 token context)
Upgraded from all-MiniLM-L6-v2 (22M params, 256 token context) per expert
review — the old model silently truncated half of longer question scenarios.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

# ChromaDB is optional — install with: pip install chromadb
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# sentence-transformers for embedding
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "staffml_questions_v2"  # Bumped version for new embedding model

# nomic-embed-text-v1.5: 137M params, 8192 token context, MTEB ~62
# Critical upgrade: old model (MiniLM, 256 tokens) truncated longer scenarios
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 768  # nomic supports 64-768 via Matryoshka; use full for quality
FALLBACK_MODEL = "all-MiniLM-L6-v2"  # If nomic fails to download

PERSIST_DIR = Path(__file__).parent.parent / "_vectordb"


# ---------------------------------------------------------------------------
# Shared embedding function
# ---------------------------------------------------------------------------

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Get or create a cached SentenceTransformer model."""
    if not HAS_ST:
        raise ImportError("sentence-transformers required: pip3 install sentence-transformers")

    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = SentenceTransformer(
                model_name, trust_remote_code=True
            )
        except Exception:
            # Fall back to MiniLM if nomic fails
            if model_name != FALLBACK_MODEL:
                _model_cache[model_name] = SentenceTransformer(FALLBACK_MODEL)
            else:
                raise

    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of texts into vectors.

    Returns numpy array of shape (len(texts), embedding_dim).
    Used by both ChromaDB and BERTopic pipelines.
    """
    model = get_embedder(model_name)
    # nomic requires "search_document: " prefix for documents
    if "nomic" in model_name:
        prefixed = [f"search_document: {t}" for t in texts]
    else:
        prefixed = texts
    return model.encode(prefixed, show_progress_bar=len(texts) > 100)


def corpus_to_texts(corpus: list[dict]) -> list[str]:
    """Convert corpus entries to embedding-ready text strings.

    Includes title, topic, scenario, and napkin math for richer embeddings.
    """
    texts = []
    for q in corpus:
        parts = [
            q.get("title", ""),
            q.get("topic", ""),
            q.get("scenario", ""),
        ]
        # Include napkin math for richer semantic content
        details = q.get("details", {})
        if details.get("napkin_math"):
            parts.append(details["napkin_math"][:500])
        texts.append(" | ".join(p for p in parts if p))
    return texts


# ---------------------------------------------------------------------------
# ChromaDB embedding engine
# ---------------------------------------------------------------------------

class QuestionEmbedder:
    """Manages the ChromaDB vector store for StaffML questions."""

    def __init__(self, persist_dir: Optional[Path] = None):
        if not HAS_CHROMA:
            raise ImportError(
                "ChromaDB is required for embedding.\n"
                "Install with: pip3 install chromadb sentence-transformers"
            )

        self.persist_dir = persist_dir or PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_corpus(self, corpus_path: Optional[Path] = None) -> int:
        """Embed all questions from corpus.json into ChromaDB.

        Uses nomic-embed-text-v1.5 for 8192-token context embeddings.
        Returns the number of questions embedded.
        """
        if corpus_path is None:
            corpus_path = Path(__file__).parent.parent / "corpus.json"

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. Run build_corpus.py first."
            )

        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

        # Clear existing
        existing = self.collection.count()
        if existing > 0:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)

        # Build texts and embed
        texts = corpus_to_texts(corpus)
        embeddings = embed_texts(texts)

        # Batch insert into ChromaDB
        batch_size = 100
        total = 0
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_embeds = embeddings[i:i + batch_size].tolist()

            ids = []
            metadatas = []

            for j, q in enumerate(batch):
                ids.append(q.get("id", f"q-{total + j}"))
                metadatas.append({
                    "track": q.get("track", "unknown"),
                    "level": q.get("level", "unknown"),
                    "topic": q.get("topic", "unknown"),
                    "title": q.get("title", "unknown"),
                    "source": q.get("source", "handwritten"),
                })

            self.collection.add(
                ids=ids,
                documents=batch_texts,
                embeddings=batch_embeds,
                metadatas=metadatas,
            )
            total += len(batch)

        return total

    def find_gaps(
        self,
        concepts: list[str],
        threshold: float = 0.5,
    ) -> list[dict]:
        """Find concepts with low question coverage."""
        if self.collection.count() == 0:
            raise RuntimeError("No questions embedded. Run embed_corpus() first.")

        # Embed concepts with the same model
        concept_embeddings = embed_texts(concepts)

        gaps = []
        for idx, concept in enumerate(concepts):
            results = self.collection.query(
                query_embeddings=[concept_embeddings[idx].tolist()],
                n_results=5,
            )

            if not results["distances"] or not results["distances"][0]:
                gaps.append({
                    "concept": concept,
                    "coverage": 0.0,
                    "nearest_questions": [],
                    "is_gap": True,
                })
                continue

            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            # Similarity = 1 - distance (per ChromaDB docs)
            distances = results["distances"][0]
            best_similarity = 1 - distances[0]

            nearest = []
            for j, (dist, meta) in enumerate(
                zip(distances[:3], results["metadatas"][0][:3])
            ):
                sim = 1 - dist
                nearest.append({
                    "title": meta.get("title", "?"),
                    "similarity": round(sim, 3),
                    "level": meta.get("level", "?"),
                    "track": meta.get("track", "?"),
                })

            gaps.append({
                "concept": concept,
                "coverage": round(best_similarity, 3),
                "nearest_questions": nearest,
                "is_gap": best_similarity < threshold,
            })

        gaps.sort(key=lambda g: g["coverage"])
        return gaps

    def check_duplicate(
        self,
        title: str,
        scenario: str,
        threshold: float = 0.85,
    ) -> tuple[bool, float]:
        """Check if a question is too similar to an existing one."""
        query_text = f"{title} | {scenario}"
        query_embed = embed_texts([query_text])

        results = self.collection.query(
            query_embeddings=query_embed.tolist(),
            n_results=1,
        )

        if not results["distances"] or not results["distances"][0]:
            return False, 0.0

        distance = results["distances"][0][0]
        similarity = 1 - distance
        return similarity >= threshold, round(similarity, 3)

    def coverage_report(self) -> dict:
        """Generate a coverage report showing question distribution."""
        if self.collection.count() == 0:
            return {"total": 0, "by_track": {}, "by_level": {}}

        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data["metadatas"]

        by_track: dict[str, int] = {}
        by_level: dict[str, int] = {}
        by_track_level: dict[str, dict[str, int]] = {}

        for meta in metadatas:
            track = meta.get("track", "unknown")
            level = meta.get("level", "unknown")

            by_track[track] = by_track.get(track, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1

            if track not in by_track_level:
                by_track_level[track] = {}
            by_track_level[track][level] = by_track_level[track].get(level, 0) + 1

        return {
            "total": len(metadatas),
            "by_track": dict(sorted(by_track.items())),
            "by_level": dict(sorted(by_level.items())),
            "by_track_level": by_track_level,
        }


# ---------------------------------------------------------------------------
# Concept extraction from TOPIC_MAP.md
# ---------------------------------------------------------------------------

def extract_concepts_from_topic_map(
    topic_map_path: Optional[Path] = None,
) -> list[str]:
    """Extract testable concepts from TOPIC_MAP.md."""
    if topic_map_path is None:
        topic_map_path = Path(__file__).parent.parent / "TOPIC_MAP.md"

    if not topic_map_path.exists():
        return []

    text = topic_map_path.read_text(encoding="utf-8")
    concepts = []

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("| ☁️") or line.startswith("| 🤖") or \
           line.startswith("| 📱") or line.startswith("| 🔬"):
            parts = line.split("|")
            if len(parts) >= 3:
                manifestation = parts[2].strip()
                if manifestation and manifestation != "Manifestation":
                    concepts.append(manifestation)

    return concepts
