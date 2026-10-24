"""Coverage numbers are computed from recorded judgments that must stay valid."""
import pytest
import yaml

from core.taxonomy_coverage import coverage_report, write_report


@pytest.fixture
def repo(tmp_path):
    concepts = tmp_path / "shared/concepts"
    (concepts / "coverage").mkdir(parents=True)
    (concepts / "ml-systems-field.yml").write_text(yaml.safe_dump({"concepts": [{"id": "mlsys.kv-cache"}, {"id": "mlsys.batching"}]}))
    (concepts / "ml-systems-framework.yml").write_text(yaml.safe_dump({"areas": [{"rows": [{"id": "llm-inference", "code": "L4.4"}]}]}))
    return tmp_path


def checklist(topics, held_out=True):
    return {"id": "test", "title": "Test checklist", "basis": "A bounded source.", "held_out": held_out, "topics": topics}


def save(root, data):
    (root / "shared/concepts/coverage/test.yml").write_text(yaml.safe_dump(data))


def test_numbers_are_computed_and_weighted_by_sources(repo):
    save(repo, checklist([
        {"topic": "KV caching", "sources": ["a", "b", "c"], "status": "covered", "concepts": ["mlsys.kv-cache"]},
        {"topic": "Paged attention", "sources": ["a"], "status": "partial", "subject": "L4.4"},
        {"topic": "Visualization", "status": "missing"},
        {"topic": "ML for systems", "status": "out-of-scope"},
    ]))
    report = write_report(repo)
    summary = report["checklists"][0]["summary"]
    assert summary["in_scope"] == 3
    assert summary["covered_pct"] == 33.3
    assert summary["homed_pct"] == 66.7
    assert summary["weighted_covered_pct"] == 60.0
    assert report["errors"] == []
    assert (repo / "books/_build/concepts/taxonomy-coverage.md").exists()


@pytest.mark.parametrize("topic,error", [
    ({"topic": "Gone", "status": "covered", "concepts": ["mlsys.removed"]}, "no longer exist: mlsys.removed"),
    ({"topic": "No concept", "status": "covered"}, "covered requires at least one concept"),
    ({"topic": "No subject", "status": "partial"}, "partial requires a subject"),
    ({"topic": "Bad subject", "status": "partial", "subject": "Z9.9"}, "unknown subject Z9.9"),
    ({"topic": "Bad status", "status": "kind of"}, "unknown status"),
])
def test_stale_or_malformed_judgments_are_reported(repo, topic, error):
    save(repo, checklist([topic]))
    assert any(error in e for e in coverage_report(repo)["errors"])


def test_checklist_must_declare_whether_it_is_held_out(repo):
    data = checklist([{"topic": "KV caching", "status": "covered", "concepts": ["mlsys.kv-cache"]}])
    del data["held_out"]
    save(repo, data)
    assert any("held_out" in e for e in coverage_report(repo)["errors"])
