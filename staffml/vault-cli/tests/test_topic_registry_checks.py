"""Checks #11 (topic in registry) and #14 (prerequisite DAG) read the live registry.

Until 2026-10-22 both read vault/taxonomy.yaml, a file that no longer existed,
so they passed without checking anything. These tests pin that they now fail.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from vault_cli.loader import LoadedQuestion
from vault_cli.models import Details, Question
from vault_cli.validator import TOPIC_REGISTRY, _check_taxonomy_dag, structural_tier


def _registry(vault_dir: Path, topics: list[dict]) -> None:
    path = vault_dir / TOPIC_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"topics": topics}))


def _lq(topic: str) -> LoadedQuestion:
    q = Question(id="cloud-9999", track="cloud", level="L4", zone="diagnosis", topic=topic, competency_area="memory",
                 bloom_level="analyze", title="T", scenario="A scenario long enough to be useful.",
                 details=Details(realistic_solution="answer."), status="draft", provenance="human")
    return LoadedQuestion(question=q, path=Path("/tmp/cloud-9999.yaml"))


def test_unknown_topic_fails(tmp_path: Path):
    _registry(tmp_path, [{"id": "kv-cache-management"}])
    checks = {f.check for f in structural_tier([_lq("not-a-topic")], tmp_path)}
    assert "topic-in-taxonomy" in checks
    checks = {f.check for f in structural_tier([_lq("kv-cache-management")], tmp_path)}
    assert "topic-in-taxonomy" not in checks


def test_prerequisite_cycle_fails_and_related_cycle_does_not(tmp_path: Path):
    _registry(tmp_path, [
        {"id": "a", "edges": [{"target": "b", "edge_type": "prerequisite"}]},
        {"id": "b", "edges": [{"target": "a", "edge_type": "prerequisite"}]},
    ])
    assert _check_taxonomy_dag(tmp_path)
    _registry(tmp_path, [
        {"id": "a", "edges": [{"target": "b", "edge_type": "related"}]},
        {"id": "b", "edges": [{"target": "a", "edge_type": "related"}]},
    ])
    assert _check_taxonomy_dag(tmp_path) == []
