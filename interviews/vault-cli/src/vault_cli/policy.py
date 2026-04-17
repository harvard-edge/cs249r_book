"""Release-policy filter predicate — SINGLE source of truth.

No consumer (paper exporter, site, D1 migration emitter) may re-implement this
logic. The import-graph CI check enforces that. See ARCHITECTURE.md §11.3 and
REVIEWS.md H-21.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def load_policy(policy_path: Path) -> dict[str, Any]:
    with policy_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{policy_path}: expected mapping at top level")
    return data


def is_published(question: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    """Predicate: does this question belong in the release under ``policy``?"""
    include = policy.get("include", {}) or {}
    allowed_status = set(include.get("status", []) or [])
    require_validated = bool(include.get("require_validated", False))

    if question.get("status") not in allowed_status:
        return False
    if require_validated and question.get("validated") is not True:
        return False
    if question.get("id") in set(policy.get("exclude_ids", []) or []):
        return False
    return question.get("topic") not in set(policy.get("exclude_topics", []) or [])


def filter_questions(
    questions: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Apply the policy predicate; return the questions that pass."""
    return [q for q in questions if is_published(q, policy)]


def policy_version(policy: Mapping[str, Any]) -> int:
    v = policy.get("policy_version")
    if not isinstance(v, int) or v < 1:
        raise ValueError(f"invalid policy_version={v!r}; expected positive integer")
    return v


__all__ = ["load_policy", "is_published", "filter_questions", "policy_version"]
