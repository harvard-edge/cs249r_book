"""Coverage accounting must not turn suggestions into reviewed coverage."""
import pytest
import yaml

from core import concept_coverage as coverage
from core.concept_graph import ConceptGraphError


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    chapter = dict(id="vol1/01_intro", volume="vol1", source=tmp_path / "books/vol1/01_intro/intro.qmd",
                   source_hash="chapter-original", sections={"sec-queue": "Waiting grows as competing requests contend for service."},
                   contract=dict(primary_concepts=["Queueing"], secondary_concepts=["Admission"], claims={"Queueing": "Waiting depends on arrivals."}))
    graph = dict(chapters={chapter["id"]: chapter}, nodes={"vol1.queue": dict(id="vol1.queue", chapter=chapter["id"], concept="Queueing")})
    field = dict(concepts=[dict(id="mlsys.queue", label="Queueing", definition="Waiting for service.")],
                 book_links=[dict(concept="mlsys.queue", book_concept="vol1.queue")])
    monkeypatch.setattr(coverage, "load_graph", lambda root: graph)
    monkeypatch.setattr(coverage, "load_field", lambda root, graph: field)
    return tmp_path, graph, field


def decision(root):
    item = coverage.coverage_report(root)["inventory"][0]
    d = dict(inventory_id=item["inventory_id"], chapter=item["chapter"], label=item["label"], role=item["role"], disposition="mapped-to", field_concepts=["mlsys.queue"],
             rationale="The passage describes waiting for service.", reviewer="test reviewer", reviewer_kind="human", reviewed_on="2026-10-21",
             evidence=[dict(section="sec-queue", quote="Waiting grows as competing requests contend for service.")])
    graph = coverage.load_graph(root)
    field = coverage.load_field(root, graph)
    d["source_hash"], d["field_hash"] = coverage.decision_hashes(graph["chapters"][item["chapter"]], item["label"], item["role"], {c["id"]: c for c in field["concepts"]}, d)
    return d


def write(root, decisions):
    path = root / "taxonomy/coverage-ledger.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(schema_version=1, decisions=decisions)))


def test_suggestions_are_not_decisions_and_all_declarations_are_counted(inventory):
    root, _, _ = inventory
    report = coverage.coverage_report(root)
    assert len(report["inventory"]) == 2
    assert report["inventory"][0]["suggestions"] == dict(label_matches=["mlsys.queue"], related_passage_links=["mlsys.queue"])
    assert report["inventory"][0]["status"] == "unreviewed"
    assert not report["gate_passed"]


def test_current_decision_only_resolves_its_declaration(inventory):
    root, _, _ = inventory
    write(root, [decision(root)])
    report = coverage.coverage_report(root)
    assert [i["status"] for i in report["inventory"]] == ["current", "unreviewed"]
    assert report["gate_passed"]  # Gate explicitly covers primary declarations only.


@pytest.mark.parametrize("change,reason", [
    (lambda g, f: g["chapters"]["vol1/01_intro"]["contract"]["claims"].update(Queueing="New claim"), "cited-section-or-declaration-changed"),
    (lambda g, f: g["chapters"]["vol1/01_intro"]["sections"].update({"sec-queue": "Different evidence."}), "manuscript-evidence-changed:sec-queue"),
    (lambda g, f: f["concepts"][0].update(definition="Different field definition."), "dependent-field-records-changed"),
])
def test_changes_reopen_review(inventory, change, reason):
    root, graph, field = inventory
    write(root, [decision(root)])
    change(graph, field)
    report = coverage.coverage_report(root)
    assert report["inventory"][0]["status"] == "stale"
    assert reason in report["inventory"][0]["stale_reasons"]
    assert not report["gate_passed"]


def test_deleted_declaration_leaves_orphaned_decision(inventory):
    root, graph, _ = inventory
    d = decision(root)
    write(root, [d])
    graph["chapters"]["vol1/01_intro"]["contract"]["primary_concepts"] = []
    report = coverage.coverage_report(root)
    assert report["orphaned_decisions"] == [d["inventory_id"]]
    assert not report["gate_passed"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(field_concepts=["unknown"]),
    lambda d: d.update(disposition="out-of-scope"),
    lambda d: d.update(evidence=[]),
    lambda d: d.update(reviewer_kind="certified"),
])
def test_invalid_decisions_fail_closed(inventory, mutate):
    root, _, _ = inventory
    d = decision(root)
    mutate(d)
    write(root, [d])
    with pytest.raises(ConceptGraphError):
        coverage.coverage_report(root)


def test_report_is_repeatable_and_empty_scope_does_not_pass(inventory):
    root, _, _ = inventory
    assert not coverage.coverage_report(root, ("vol4",))["gate_passed"]
    coverage.write_coverage_report(root)
    path = root / "books/_build/concepts/coverage-audit.json"
    before = path.read_bytes()
    coverage.write_coverage_report(root)
    assert path.read_bytes() == before


def test_unrelated_edits_do_not_invalidate_targeted_decision(inventory):
    root, graph, field = inventory
    write(root, [decision(root)])
    graph["chapters"]["vol1/01_intro"].update(source_hash="unrelated chapter edit")
    graph["chapters"]["vol1/01_intro"]["sections"]["sec-other"] = "Unrelated section."
    field["concepts"].append(dict(id="mlsys.other", label="Other", definition="Unrelated."))
    assert coverage.coverage_report(root)["gate_passed"]


def test_ai_decision_is_visible_but_cannot_pass_human_gate(inventory):
    root, _, _ = inventory
    d = decision(root)
    d["reviewer_kind"] = "ai"
    write(root, [d])
    report = coverage.coverage_report(root)
    assert report["inventory"][0]["status"] == "current"
    assert report["by_volume"]["vol1"]["reviewer_kinds"] == {"ai": 1}
    assert not report["gate_passed"]


def test_removed_field_target_reopens_instead_of_crashing(inventory):
    root, _, field = inventory
    write(root, [decision(root)])
    field["concepts"].clear()
    report = coverage.coverage_report(root)
    assert "field-target-removed" in report["inventory"][0]["stale_reasons"]
    assert not report["gate_passed"]


def test_yaml_date_serializes_and_unknown_key_is_rejected(inventory):
    from datetime import date
    root, _, _ = inventory
    d = decision(root)
    d["reviewed_on"] = date(2026, 10, 21)
    write(root, [d])
    report = coverage.write_coverage_report(root)
    assert report["inventory"][0]["decision"]["reviewed_on"] == "2026-10-21"
    d["field_concept"] = "typo"
    write(root, [d])
    with pytest.raises(ConceptGraphError, match="unknown decision keys"):
        coverage.coverage_report(root)


def test_other_volume_orphan_does_not_block_selected_volume(inventory):
    root, _, _ = inventory
    d = decision(root)
    orphan = dict(d, inventory_id="vol3/deleted:123", chapter="vol3/deleted")
    write(root, [d, orphan])
    assert coverage.coverage_report(root, ("vol1",))["gate_passed"]
    assert not coverage.coverage_report(root)["gate_passed"]


def test_trivial_evidence_quote_is_rejected(inventory):
    root, _, _ = inventory
    d = decision(root)
    d["evidence"][0]["quote"] = "the"
    write(root, [d])
    with pytest.raises(ConceptGraphError, match="quote too short"):
        coverage.coverage_report(root)


def test_untargeted_decision_reopens_after_definition_change(inventory):
    root, graph, field = inventory
    d = decision(root)
    d.update(disposition="candidate-new", field_concepts=[])
    d["source_hash"], d["field_hash"] = coverage.decision_hashes(graph["chapters"][d["chapter"]], d["label"], d["role"], {c["id"]:c for c in field["concepts"]}, d)
    write(root, [d])
    field["concepts"][0]["definition"] = "A broadened definition."
    report = coverage.coverage_report(root)
    assert report["inventory"][0]["status"] == "stale"


def test_future_review_date_cannot_pass_gate(inventory):
    from datetime import date
    root, _, _ = inventory
    d = decision(root)
    d["reviewed_on"] = date.max
    write(root, [d])
    with pytest.raises(ConceptGraphError, match="nonfuture ISO"):
        coverage.coverage_report(root)
