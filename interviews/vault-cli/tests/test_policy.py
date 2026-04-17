"""Tests for the release-policy filter.

Critical invariant (REVIEWS.md H-21): every exporter must call the SAME
policy.filter_questions function. This test is the runtime analogue; CI's
import-graph check is the static-analysis complement.
"""

from __future__ import annotations

from vault_cli.policy import filter_questions, is_published, policy_version


def _policy() -> dict:
    return {
        "policy_version": 1,
        "include": {"status": ["published"], "require_validated": False},
        "exclude_topics": [],
        "exclude_ids": [],
    }


def test_published_passes() -> None:
    q = {"id": "x", "status": "published", "topic": "t"}
    assert is_published(q, _policy()) is True


def test_draft_rejected() -> None:
    q = {"id": "x", "status": "draft", "topic": "t"}
    assert is_published(q, _policy()) is False


def test_exclude_id_wins() -> None:
    p = _policy()
    p["exclude_ids"] = ["x"]
    q = {"id": "x", "status": "published", "topic": "t"}
    assert is_published(q, p) is False


def test_exclude_topic_wins() -> None:
    p = _policy()
    p["exclude_topics"] = ["t"]
    q = {"id": "x", "status": "published", "topic": "t"}
    assert is_published(q, p) is False


def test_filter_returns_published_only() -> None:
    qs = [
        {"id": "a", "status": "published", "topic": "t"},
        {"id": "b", "status": "draft", "topic": "t"},
        {"id": "c", "status": "deprecated", "topic": "t"},
    ]
    out = filter_questions(qs, _policy())
    assert {q["id"] for q in out} == {"a"}


def test_policy_version_returns_int() -> None:
    assert policy_version(_policy()) == 1
