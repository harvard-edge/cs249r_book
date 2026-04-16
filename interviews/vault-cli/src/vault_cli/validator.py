"""Invariant checker.

Implements the tiered checks in ARCHITECTURE.md §5. Fast-tier checks run in
the pre-commit hook; structural-tier checks run in CI; slow-tier checks run
nightly.

This module is the engine; tier selection and reporting are in
``commands/check.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vault_cli.loader import LoadedQuestion
from vault_cli.paths import is_lowercase, vault_questions_root
from vault_cli.yaml_io import load_file


@dataclass(frozen=True)
class InvariantFailure:
    tier: str
    check: str
    question_id: str | None
    path: Path | None
    message: str


def _fail(tier: str, check: str, *, message: str, qid: str | None = None, path: Path | None = None) -> InvariantFailure:
    return InvariantFailure(tier=tier, check=check, question_id=qid, path=path, message=message)


def fast_tier(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    """Fast-tier invariants — pre-commit-hook speed."""
    failures: list[InvariantFailure] = []

    # Check #2: unique IDs across published+draft
    ids = Counter(q.id for q in loaded)
    for qid, n in ids.items():
        if n > 1:
            failures.append(_fail("fast", "unique-id", qid=qid, message=f"ID appears {n} times"))

    # Check #4: path components lowercase
    # Check #5: path components enum-valid — already enforced by paths.classification_from_path
    #           (raises ValueError for non-enum values), so here just double-check lowercase.
    root = vault_questions_root(vault_dir)
    for lq in loaded:
        rel = lq.path.relative_to(root)
        for component in rel.parts[:3]:
            if not is_lowercase(component):
                failures.append(
                    _fail(
                        "fast",
                        "path-lowercase",
                        qid=lq.id,
                        path=lq.path,
                        message=f"path component {component!r} is not lowercase",
                    )
                )

    return failures


def _load_yaml_set(vault_dir: Path, filename: str, key: str) -> set[str]:
    """Load ``{vault_dir}/{filename}`` and extract IDs from its top-level ``key`` list."""
    path = vault_dir / filename
    if not path.exists():
        return set()
    data = load_file(path)
    if not isinstance(data, dict):
        return set()
    items = data.get(key, []) or []
    return {item["id"] if isinstance(item, dict) and "id" in item else item for item in items if item}


def structural_tier(
    loaded: list[LoadedQuestion],
    vault_dir: Path,
) -> list[InvariantFailure]:
    """Structural invariants — CI-tier checks."""
    failures: list[InvariantFailure] = []

    # #11: every `topic` exists in taxonomy.yaml
    known_topics = _load_yaml_set(vault_dir, "taxonomy.yaml", "topics")
    if known_topics:
        for lq in loaded:
            if lq.question.topic not in known_topics:
                failures.append(
                    _fail(
                        "structural",
                        "topic-in-taxonomy",
                        qid=lq.id,
                        path=lq.path,
                        message=f"topic {lq.question.topic!r} not found in taxonomy.yaml",
                    )
                )

    # #12: every chain.id exists in chains.yaml
    # #13: chain positions form contiguous [1..N]
    known_chains = _load_yaml_set(vault_dir, "chains.yaml", "chains")
    chain_members: dict[str, list[int]] = {}
    for lq in loaded:
        c = lq.question.chain
        if c is None:
            continue
        if known_chains and c.id not in known_chains:
            failures.append(
                _fail(
                    "structural",
                    "chain-ref-exists",
                    qid=lq.id,
                    path=lq.path,
                    message=f"chain {c.id!r} not found in chains.yaml",
                )
            )
        chain_members.setdefault(c.id, []).append(c.position)
    for chain_id, positions in chain_members.items():
        positions.sort()
        expected = list(range(1, len(positions) + 1))
        if positions != expected:
            failures.append(
                _fail(
                    "structural",
                    "chain-positions-contiguous",
                    qid=None,
                    message=f"chain {chain_id!r} positions {positions} not contiguous {expected}",
                )
            )

    # #18: provenance metadata consistency
    for lq in loaded:
        if lq.question.provenance.value != "human" and lq.question.generation_meta is None:
            failures.append(
                _fail(
                    "structural",
                    "provenance-meta",
                    qid=lq.id,
                    path=lq.path,
                    message="non-human provenance requires generation_meta",
                )
            )

    return failures


def run_all(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    return fast_tier(loaded, vault_dir) + structural_tier(loaded, vault_dir)


__all__ = ["InvariantFailure", "fast_tier", "structural_tier", "run_all"]
