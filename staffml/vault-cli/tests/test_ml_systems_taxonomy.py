"""ML Systems Taxonomy tags (taxonomy_subject, taxonomy_concepts), added 2026-10.

The model checks the identity format; the structural tier checks that every
tag names a subject or concept in the repo's taxonomy; the exporter, compiler,
and content hash carry the tags through.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from vault_cli.hashing import content_hash
from vault_cli.legacy_export import emit_legacy_corpus
from vault_cli.loader import LoadedQuestion
from vault_cli.models import Details, Question
from vault_cli.validator import _check_ml_systems_taxonomy


def _question(**tags) -> Question:
    return Question(
        id="cloud-9999", track="cloud", level="L4", zone="diagnosis", topic="kv-cache-management",
        competency_area="memory", bloom_level="analyze", title="T", scenario="A scenario long enough to be useful.",
        details=Details(realistic_solution="answer."), status="published", provenance="human", **tags,
    )


def _lq(**tags) -> LoadedQuestion:
    return LoadedQuestion(question=_question(**tags), path=Path("/tmp/cloud-9999.yaml"))


def test_tags_are_optional():
    q = _question()
    assert q.taxonomy_subject is None and q.taxonomy_concepts is None


@pytest.mark.parametrize("tags", [
    {"taxonomy_subject": "LLM Inference"},
    {"taxonomy_subject": "llm-inference", "taxonomy_concepts": ["kv-cache"]},
    {"taxonomy_subject": "llm-inference", "taxonomy_concepts": ["mlsys.kv-cache", "mlsys.kv-cache"]},
    {"taxonomy_subject": "llm-inference", "taxonomy_concepts": ["mlsys.a", "mlsys.b", "mlsys.c", "mlsys.d"]},
])
def test_malformed_tags_are_rejected(tags):
    with pytest.raises(ValueError):
        _question(**tags)


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    concepts = tmp_path / "taxonomy"
    concepts.mkdir(parents=True)
    (concepts / "ml-systems-framework.yml").write_text(yaml.safe_dump({"areas": [{"rows": [{"id": "llm-inference"}]}]}))
    (concepts / "ml-systems-field.yml").write_text(yaml.safe_dump({
        "concepts": [{"id": "mlsys.kv-cache"}],
        "retired": [{"id": "mlsys.old", "replaced_by": "mlsys.kv-cache"}],
    }))
    vault = tmp_path / "staffml" / "vault"
    vault.mkdir(parents=True)
    return vault


def test_structural_check_accepts_known_identities(vault_dir: Path):
    assert _check_ml_systems_taxonomy([_lq(taxonomy_subject="llm-inference", taxonomy_concepts=["mlsys.kv-cache"])], vault_dir) == []


def test_structural_check_rejects_unknown_and_retired_identities(vault_dir: Path):
    failures = _check_ml_systems_taxonomy([
        _lq(taxonomy_subject="no-such-subject"),
        _lq(taxonomy_subject="llm-inference", taxonomy_concepts=["mlsys.old"]),
        _lq(taxonomy_concepts=["mlsys.kv-cache"]),
    ], vault_dir)
    messages = " | ".join(f.message for f in failures)
    assert "no-such-subject" in messages
    assert "replaced by 'mlsys.kv-cache'" in messages
    assert "requires a taxonomy_subject" in messages


def test_structural_check_skips_without_taxonomy(tmp_path: Path):
    assert _check_ml_systems_taxonomy([_lq(taxonomy_subject="anything")], tmp_path / "staffml" / "vault") == []


def test_tags_reach_the_corpus_and_the_hash(tmp_path: Path):
    tagged = _lq(taxonomy_subject="llm-inference", taxonomy_concepts=["mlsys.kv-cache"])
    (tmp_path / "release-policy.yaml").write_text("policy_version: 1\ninclude: {status: [published], require_validated: false}\n")
    out = tmp_path / "corpus.json"
    emit_legacy_corpus(tmp_path, [tagged], out)
    item = json.loads(out.read_text())[0]
    assert item["taxonomy_subject"] == "llm-inference"
    assert item["taxonomy_concepts"] == ["mlsys.kv-cache"]
    untagged = _question().model_dump(mode="json")
    assert content_hash(tagged.question.model_dump(mode="json")) != content_hash(untagged)


def test_compiler_stores_subject_and_ordered_concepts(tmp_path: Path):
    from vault_cli.compiler import DDL, _compile_questions

    conn = sqlite3.connect(tmp_path / "v.db")
    conn.executescript(DDL)
    _compile_questions(conn, [_lq(taxonomy_subject="llm-inference", taxonomy_concepts=["mlsys.kv-cache", "mlsys.paged-kv"])])
    assert conn.execute("SELECT taxonomy_subject FROM questions").fetchone()[0] == "llm-inference"
    rows = conn.execute("SELECT concept_id, position FROM question_concepts ORDER BY position").fetchall()
    assert rows == [("mlsys.kv-cache", 0), ("mlsys.paged-kv", 1)]
