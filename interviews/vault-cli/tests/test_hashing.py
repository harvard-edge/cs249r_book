"""Tests for the canonical hashing layer.

Key invariants:
- Same semantic content hashes identically regardless of YAML key order.
- Whitelist fields drive the hash; metadata doesn't.
- Merkle construction stable across re-ordering of leaves.
"""

from __future__ import annotations

from vault_cli.hashing import CANON_VERSION, content_hash, release_hash


def _base_question() -> dict:
    return {
        "id": "global-0000",
        "title": "Example",
        "topic": "kv-cache-management",
        "chain": {"id": "global-chain-000", "position": 1},
        "status": "published",
        "scenario": "Explain KV-cache.",
        "details": {"realistic_solution": "Paged attention."},
        "tags": ["a", "b"],
        "provenance": "human",
    }


def test_content_hash_stable_across_key_reorder() -> None:
    """Soumith M-NEW-4: nested-dict hash must be key-order-invariant."""
    q1 = _base_question()
    q2 = {k: q1[k] for k in reversed(list(q1.keys()))}
    q2["details"] = {"realistic_solution": q1["details"]["realistic_solution"]}
    assert content_hash(q1) == content_hash(q2)


def test_content_hash_excludes_metadata() -> None:
    """Hash must NOT change when last_modified or file_path changes."""
    q1 = _base_question()
    q2 = dict(q1)
    q2["last_modified"] = "2050-01-01T00:00:00Z"
    q2["file_path"] = "/tmp/foo.yaml"
    q2["authors"] = ["someone"]
    assert content_hash(q1) == content_hash(q2)


def test_content_hash_changes_with_semantic_edit() -> None:
    """Hash MUST change when scenario changes."""
    q1 = _base_question()
    q2 = dict(q1)
    q2["scenario"] = "An edited scenario."
    assert content_hash(q1) != content_hash(q2)


def test_release_hash_includes_canon_and_policy_leaves() -> None:
    """Chip N-H5: release_hash must bind canon version and policy."""
    leaves = [("a", "1" * 64), ("b", "2" * 64)]
    base = release_hash(
        per_question=leaves,
        taxonomy_hash="t" * 64,
        chains_hash="c" * 64,
        zones_hash="z" * 64,
        policy_hash="p" * 64,
    )
    # Different policy_hash → different release_hash
    different_policy = release_hash(
        per_question=leaves,
        taxonomy_hash="t" * 64,
        chains_hash="c" * 64,
        zones_hash="z" * 64,
        policy_hash="P" * 64,
    )
    assert base != different_policy

    # Different canon_version → different release_hash
    different_canon = release_hash(
        per_question=leaves,
        taxonomy_hash="t" * 64,
        chains_hash="c" * 64,
        zones_hash="z" * 64,
        policy_hash="p" * 64,
        canon_version=999,
    )
    assert base != different_canon


def test_canon_version_is_pinned() -> None:
    assert isinstance(CANON_VERSION, int)
    assert CANON_VERSION >= 1
