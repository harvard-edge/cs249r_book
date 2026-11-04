"""Editorial corrections preserve evidence, identity, and review boundaries."""
from copy import deepcopy

import pytest

from core.concept_graph import ConceptGraphError
from core.taxonomy_editorial import apply_review, advance_provenance


@pytest.fixture
def example():
    doc = {"version": "0.4-draft", "concepts": [{"id": "c", "definition": "Original", "aliases": ["wrong"]}],
           "relationships": [{"id": "edge", "source": "c", "target": "other"}]}
    review = {"reviewer": "Editorial assistant", "reviewer_kind": "model", "decision": "accept", "changes": [{
        "collection": "concepts", "id": "c", "before": deepcopy(doc["concepts"][0]),
        "fields": {"definition": "Corrected meaning", "aliases": []},
        "rationale": "The original alias names a different mechanism.",
        "sources": [{"url": "https://example.org/source", "locator": "Definition", "supports": "The distinction."}],
        "incident_relationships": ["edge"],
    }]}
    return doc, review


def test_replay_keeps_input_and_alias_history(example):
    doc, review = example
    after = apply_review(doc, review)
    assert doc["concepts"][0]["aliases"] == ["wrong"]
    assert after["concepts"][0]["aliases"] == []
    state = {"concepts": {}, "grandfathered_draft_alias_exceptions": {},
             "pending_campaign_ids": [], "pending_pre_evaluation_override": False}
    advance_provenance(state, {"campaign_id": "review-1", "replayed_patch": review})
    assert state["concepts"]["c"]["removed_aliases"]["wrong"] == review["changes"][0]["rationale"]


@pytest.mark.parametrize("defect", ["stale", "source", "edges", "identity", "new_alias", "reject", "release"])
def test_rejects_incomplete_or_out_of_scope_changes(example, defect):
    doc, review = example
    change = review["changes"][0]
    if defect == "stale": change["before"]["definition"] = "Different baseline"
    elif defect == "source": change["sources"] = []
    elif defect == "edges": change["incident_relationships"] = []
    elif defect == "identity": change["fields"]["id"] = "replacement"
    elif defect == "new_alias": change["fields"]["aliases"] = ["unreviewed"]
    elif defect == "reject": review["decision"] = "reject"
    else: doc["version"] = "1.0"
    original = deepcopy(doc)
    with pytest.raises(ConceptGraphError):
        apply_review(doc, review)
    assert doc == original


@pytest.fixture
def installer(tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools/audit/review_taxonomy.py"
    spec = importlib.util.spec_from_file_location("review_taxonomy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "taxonomy").mkdir()
    before = {name: b"original" for name in module.FILES}
    after = {name: b"candidate" for name in module.FILES}
    for name, raw in before.items():
        (tmp_path / "taxonomy" / name).write_bytes(raw)
    return module, tmp_path, before, after


def test_install_restores_both_files_on_failed_validation(installer, monkeypatch):
    module, root, before, after = installer
    def fail(_):
        raise ValueError("invalid provenance")
    monkeypatch.setattr(module.schema, "validate_term_provenance_sidecar", fail)
    with pytest.raises(ValueError, match="invalid provenance"):
        module.install(root, before, after)
    assert {name: (root / "taxonomy" / name).read_bytes() for name in before} == before


def test_install_rejects_concurrent_edit_without_overwriting(installer):
    module, root, before, after = installer
    changed = root / "taxonomy" / module.FILES[0]
    changed.write_bytes(b"another editor")
    with pytest.raises(ValueError, match="Source changed"):
        module.install(root, before, after)
    assert changed.read_bytes() == b"another editor"
    assert (root / "taxonomy" / module.FILES[1]).read_bytes() == b"original"


def relationship_review(doc, operation="update", fields=None):
    return {"reviewer": "Editorial assistant", "reviewer_kind": "model", "decision": "accept", "changes": [{
        "collection": "relationships", "id": "edge", "operation": operation,
        "before": deepcopy(doc["relationships"][0]), "fields": fields or {},
        "rationale": "The source supports association without the claimed dependency.",
        "sources": [{"url": "https://example.org/source", "locator": "Scope", "supports": "The distinction."}],
        "incident_relationships": [],
    }]}


def test_explicit_relationship_removal_is_replayable(example):
    doc, _ = example
    review = relationship_review(doc, "remove")
    after = apply_review(doc, review)
    assert after["relationships"] == []
    assert len(doc["relationships"]) == 1
    with pytest.raises(ConceptGraphError, match="unknown or duplicate"):
        apply_review(after, review)


def test_retyping_requires_matching_controlled_kind(example):
    doc, _ = example
    doc["relations"] = [{"id": "uses", "kind": "dependency"}]
    review = relationship_review(doc, fields={"relation": "uses", "kind": "dependency"})
    assert apply_review(doc, review)["relationships"][0]["relation"] == "uses"
    review["changes"][0]["fields"]["kind"] = "failure"
    with pytest.raises(ConceptGraphError, match="controlled vocabulary"):
        apply_review(doc, review)


def test_relationship_removal_cannot_hide_changes_or_remove_concepts(example):
    doc, review = example
    review["changes"][0]["operation"] = "remove"
    with pytest.raises(ConceptGraphError, match="only relationships"):
        apply_review(doc, review)
    removal = relationship_review(doc, "remove", {"conditions": "unreviewed"})
    with pytest.raises(ConceptGraphError, match="fields must be empty"):
        apply_review(doc, removal)


def test_addition_preserves_identity_and_requires_impact(example):
    doc, review = example
    change = review['changes'][0]
    change.update(operation='add', id='new', before=None,
                  fields={'id': 'new', 'definition': 'A distinct mechanism'}, incident_relationships=[])
    with pytest.raises(ConceptGraphError, match='impact'):
        apply_review(doc, review)
    change['impact'] = {k: 'Explicitly reviewed impact on this dimension.' for k in ('classification', 'assignments', 'relationships')}
    assert apply_review(doc, review)['concepts'][-1]['id'] == 'new'
    doc['retired'] = [{'id': 'new'}]
    with pytest.raises(ConceptGraphError, match='cannot be reused'):
        apply_review(doc, review)


def test_retirement_requires_explicit_edge_repair(example):
    doc, review = example
    doc['concepts'].append({'id': 'other', 'label': 'Replacement'})
    change = review['changes'][0]
    change.update(operation='retire', fields={'replaced_by': 'other'},
                  impact={k: 'Explicitly reviewed impact on this dimension.' for k in ('classification', 'assignments', 'relationships')})
    with pytest.raises(ConceptGraphError, match='incident relationships'):
        apply_review(doc, review)
    review['changes'] += relationship_review(doc, 'remove')['changes']
    result = apply_review(doc, review)
    assert [c['id'] for c in result['concepts']] == ['other']
    assert result['retired'][0]['replaced_by'] == 'other'
    assert len(doc['concepts']) == 2


def test_draft_release_transition_cannot_claim_reviewed_release(example):
    doc, review = example
    review['changes'] = []
    review['release'] = {'from': doc['version'], 'to': '0.5-draft', 'rationale': 'Freeze the reviewed editorial changes for feedback.'}
    assert apply_review(doc, review)['version'] == '0.5-draft'
    for invalid in ('0.5', '0.3-draft', '0.4.0-draft'):
        review['release']['to'] = invalid
        with pytest.raises(ConceptGraphError):
            apply_review(doc, review)


def test_entry_term_admission_requires_classification_and_preserves_evidence(example):
    doc, review = example
    change = review['changes'][0]
    change['fields'] = {'aliases': ['wrong', 'specific method']}
    change['terms'] = {'specific method': {'kind': 'narrower', 'rationale': 'A specific method within the broader mechanism.'}}
    after = apply_review(doc, review)
    assert after['concepts'][0]['aliases'][-1] == 'specific method'
    state = {'concepts': {}, 'grandfathered_draft_alias_exceptions': {}, 'pending_campaign_ids': [], 'pending_pre_evaluation_override': False}
    advance_provenance(state, {'campaign_id': 'term-review', 'replayed_patch': review})
    assert state['concepts']['c']['entry_term_decisions']['specific method']['sources'] == change['sources']
    change['terms']['specific method']['kind'] = 'guessed'
    with pytest.raises(ConceptGraphError, match='unknown entry-term'):
        apply_review(doc, review)
