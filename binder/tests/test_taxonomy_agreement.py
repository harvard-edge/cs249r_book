"""Agreement numbers are recomputed from recorded tags that must stay valid."""
from pathlib import Path
import pytest
import yaml

from core.taxonomy_agreement import agreement_report, write_report

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path):
    concepts = tmp_path / "taxonomy"
    (concepts / "agreement").mkdir(parents=True)
    (concepts / "ml-systems-taxonomy.yml").symlink_to(REPO_ROOT / "taxonomy/ml-systems-taxonomy.yml")
    (tmp_path / "books").symlink_to(REPO_ROOT / "books")
    return tmp_path


def record(tags_a, tags_b):
    return {"id": "test", "title": "Test sample", "basis": "A fixed sample.",
            "items": [{"n": n} for n in range(1, len(tags_a) + 1)],
            "runs": [{"id": "run-1", "held_out": True, "taggers": {"a": tags_a, "b": tags_b}}]}


def save(root, data):
    (root / "taxonomy/agreement/test.yml").write_text(yaml.safe_dump(data))


def test_scores_are_computed_from_tags(repo):
    a = [{"n": 1, "subject": "llm-inference", "concepts": ["mlsys.kv-cache"]},
         {"n": 2, "subject": "request-scheduling", "concepts": ["mlsys.batching"]},
         {"n": 3, "subject": "model-compression", "concepts": ["mlsys.quantization"]},
         {"n": 4, "subject": "llm-inference", "concepts": ["mlsys.kv-cache", "mlsys.batching"]}]
    b = [{"n": 1, "subject": "llm-inference", "concepts": ["mlsys.kv-cache"]},
         {"n": 2, "subject": "request-scheduling", "concepts": ["mlsys.batching"]},
         {"n": 3, "subject": "llm-inference", "concepts": ["mlsys.kv-cache"]},
         {"n": 4, "subject": "llm-inference", "concepts": ["mlsys.kv-cache"]}]
    save(repo, record(a, b))
    report = write_report(repo)
    scores = report["records"][0]["runs"][0]["scores"]
    assert report["errors"] == []
    assert scores["subject_pct"] == 75.0
    assert scores["area_pct"] == 75.0
    assert scores["primary_concept_pct"] == 75.0
    assert scores["same_concept_set_pct"] == 50.0
    assert scores["compatible_pct"] == 75.0
    assert (repo / "books/_build/concepts/taxonomy-agreement.json").exists()


def test_retired_identities_and_missing_items_are_errors(repo):
    a = [{"n": 1, "subject": "llm-inference", "concepts": ["mlsys.retired"]},
         {"n": 2, "subject": "gone-subject", "concepts": ["mlsys.kv-cache", "mlsys.kv-cache"]}]
    b = [{"n": 1, "subject": "llm-inference", "concepts": ["mlsys.kv-cache"]}]
    data = record(a, b)
    del data["runs"][0]["held_out"]
    save(repo, data)
    errors = agreement_report(repo)["errors"]
    assert any("no longer exist" in e for e in errors)
    assert any("unknown subject gone-subject" in e for e in errors)
    assert any("every sample item exactly once" in e for e in errors)
    assert any("duplicate concept tags" in e for e in errors)
    assert any("held_out true or false" in e for e in errors)
