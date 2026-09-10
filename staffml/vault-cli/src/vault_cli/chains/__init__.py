"""Chain audit and rescue suggestions.

Three layers:
  - embeddings: cached sentence-transformer embeddings per question
  - audit: orphans, position drift, stale-registry detection
  - rescue: ranked merge candidates for orphan singletons

Embedding-only by design — no LLM in the hot path. The structural
constraints (single-topic, Bloom-monotonic, ≥2 members) are enforced via
the existing validator; this module ranks candidates *within* those rules.
"""
