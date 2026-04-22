"""Canonical hashing for content_hash and release_hash.

Implements ARCHITECTURE.md §3.5. The hashes are over *inputs* (canonicalized
JSON of whitelisted semantic fields), never over SQLite binary. This is what
makes the corpus academically citable and reproducible.

Canonicalization version ``CANON_VERSION`` is pinned here; bumping it requires
co-versioning schema and is recorded as an `__canon_version__` Merkle leaf so
two releases with the same source but different canonicalizers produce
distinct release_hashes by construction.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

CANON_VERSION = 2

# Fields included in per-question content_hash. Metadata fields that can change
# without altering meaning (last_modified, file_path, authors list ordering,
# prompt_cost_usd, review stamps) are deliberately excluded.
#
# CANON_VERSION=2 (2026-04-21) adds track/level/zone/competency_area/bloom_level/phase:
# these classification axes live in the YAML body in schema v1.0. Under v0.1
# they lived in the filesystem path and were implicitly part of the merkle
# tree via path-derived leaves, but in v1.0 they must be whitelisted explicitly
# so reclassifications produce a visibly different release_hash.
# `chain` (singular) replaced by `chains` (plural) per schema v1.0.
WHITELIST_TOP: frozenset[str] = frozenset({
    "id", "title",
    "track", "level", "zone",           # v1.0: classification now in YAML
    "topic", "competency_area", "bloom_level", "phase",
    "chains",                            # v1.0: plural chain refs
    "status", "scenario", "details", "tags", "provenance",
})
WHITELIST_GEN_META: frozenset[str] = frozenset({"model", "prompt_hash"})


def _normalize_strings(obj: Any) -> Any:
    """Recursively NFC-normalize + LF-normalize all string values."""
    if isinstance(obj, str):
        nfc = unicodedata.normalize("NFC", obj)
        return nfc.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if isinstance(obj, Mapping):
        return {k: _normalize_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_strings(v) for v in obj]
    return obj


def _select(q: Mapping[str, Any]) -> dict[str, Any]:
    """Extract whitelisted fields; collapse generation_meta to its subset."""
    out: dict[str, Any] = {}
    for key in WHITELIST_TOP:
        if key in q and q[key] is not None:
            out[key] = q[key]
    gm = q.get("generation_meta")
    if isinstance(gm, Mapping):
        gm_subset = {k: gm[k] for k in WHITELIST_GEN_META if gm.get(k) is not None}
        if gm_subset:
            out["generation_meta"] = gm_subset
    return out


def _canonical_bytes(payload: Any) -> bytes:
    """Serialize normalized payload to canonical JSON bytes.

    ``sort_keys=True`` recurses into nested dicts (stdlib behavior).
    ``separators`` removes whitespace. ``ensure_ascii=False`` keeps UTF-8.
    """
    normalized = _normalize_strings(payload)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def content_hash(question: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON over whitelisted semantic fields."""
    return hashlib.sha256(_canonical_bytes(_select(question))).hexdigest()


def hash_of_canonical_yaml(data: Any) -> str:
    """SHA-256 of canonical JSON of arbitrary loaded-YAML data.

    Used for taxonomy/chains/zones/policy Merkle leaves.
    """
    return hashlib.sha256(_canonical_bytes(data)).hexdigest()


def release_hash(
    *,
    per_question: Iterable[tuple[str, str]],  # (id, content_hash)
    taxonomy_hash: str,
    chains_hash: str,
    zones_hash: str,
    policy_hash: str,
    canon_version: int = CANON_VERSION,
) -> str:
    """Merkle root over per-question content_hashes plus structural + policy leaves."""
    leaves: list[tuple[str, str]] = sorted(per_question)
    leaves.append(("__taxonomy__", taxonomy_hash))
    leaves.append(("__chains__", chains_hash))
    leaves.append(("__zones__", zones_hash))
    leaves.append(("__policy__", policy_hash))
    leaves.append(
        ("__canon_version__", hashlib.sha256(f"canon-v{canon_version}".encode()).hexdigest())
    )
    body = b"\n".join(f"{leaf_id}:{leaf_hash}".encode() for leaf_id, leaf_hash in leaves)
    return hashlib.sha256(body).hexdigest()


__all__ = ["CANON_VERSION", "content_hash", "hash_of_canonical_yaml", "release_hash"]
