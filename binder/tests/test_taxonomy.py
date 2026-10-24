"""Authored hierarchy and navigation collections must retain distinct meanings."""

from copy import deepcopy

import pytest
import yaml

from core.concept_graph import ConceptGraphError
from core.taxonomy import load_taxonomy, taxonomy_artifacts


@pytest.fixture
def example(tmp_path):
    field = {"families": [{"id": "resources"}], "concepts": [
        {"id": "mlsys.quantization", "family": "resources", "label": "Quantization", "definition": "Representing values with a restricted set.", "aliases": ["Reduced precision representation"]},
        {"id": "mlsys.memory", "family": "resources", "label": "Memory capacity", "definition": "Storage available for resident state."}],
        "relationships": [{"source": "mlsys.quantization", "target": "mlsys.memory", "predicate": "changes"}]}
    model = {"schema_version": 1, "id": "mlsys.taxonomy", "title": "ML systems taxonomy", "scope": "A declared scope.", "status": "editorial-draft",
             "sources": [{"id": "skos", "title": "SKOS", "url": "https://www.w3.org/TR/skos-reference/", "role": "Structural design reference."}],
             "topics": [{"id": "mlsys.topic.compression", "label": "Compression", "definition": "Reducing representation size.", "status": "draft"},
                        {"id": "mlsys.topic.representation", "label": "Representation", "definition": "Encoding information.", "status": "draft"}],
             "top_concepts": ["mlsys.topic.compression", "mlsys.topic.representation"],
             "hierarchy": [{"narrower": "mlsys.quantization", "broader": "mlsys.topic.compression", "rationale": "Quantization restricts value representation.", "status": "draft"},
                           {"narrower": "mlsys.quantization", "broader": "mlsys.topic.representation", "rationale": "It is a representation method.", "status": "draft"}],
             "collections": [{"id": "mlsys.collection.resources", "label": "Resource considerations", "scope_note": "A navigation grouping.", "family": "resources"}]}
    return tmp_path, field, model


def save(root, model):
    path = root / "shared/concepts/ml-systems-taxonomy.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(model, sort_keys=False))


def test_optional_absent_taxonomy_is_noop(example):
    root, field, _ = example
    assert load_taxonomy(root, field) is None
    assert taxonomy_artifacts(root, field, None) == {}


def test_multi_parent_hierarchy_preserved_without_inferred_membership(example):
    root, field, model = example
    save(root, model)
    taxonomy = load_taxonomy(root, field)
    assert taxonomy["hierarchy"] == model["hierarchy"]
    assert len(taxonomy["roots"]) == 2
    assert taxonomy["summary"]["classified_field_concepts"] == 1
    # Memory is in a navigation collection but has no asserted broader concept.
    assert all(e["narrower"] != "mlsys.memory" for e in taxonomy["hierarchy"])
    assert field["relationships"][0]["predicate"] == "changes"


@pytest.mark.parametrize("mutation, error", [
    (lambda m: m["topics"].append(deepcopy(m["topics"][0])), "duplicate topics ID"),
    (lambda m: m["topics"][0].update(id="mlsys.quantization"), "overlap field"),
    (lambda m: m["topics"][0].update(label="QUANTIZATION"), "duplicate concept label"),
    (lambda m: m["topics"][0].update(definition=" "), "requires definition"),
    (lambda m: m["topics"][0].update(status="reviewed"), "draft status"),
    (lambda m: m["sources"][0].update(url="javascript:alert(1)"), "unsafe source URL"),
    (lambda m: m["hierarchy"][0].update(narrower="mlsys.deleted"), "endpoints"),
    (lambda m: m["hierarchy"][0].update(broader="mlsys.memory"), "endpoints"),
    (lambda m: m["hierarchy"].append(deepcopy(m["hierarchy"][0])), "duplicate hierarchy"),
    (lambda m: m["hierarchy"][0].update(rationale=""), "requires rationale"),
    (lambda m: m["hierarchy"][0].update(narrower="mlsys.topic.compression"), "self hierarchy"),
    (lambda m: m["collections"].clear(), "cover every field family"),
    (lambda m: m["collections"][0].update(family="unknown"), "unknown or duplicate collection family"),
    (lambda m: m["collections"][0].update(members=["mlsys.memory"]), "must derive"),
    (lambda m: m.update(top_concepts=["mlsys.quantization"]), "root topic IDs"),
])
def test_invalid_assertions_fail_before_generation(example, mutation, error):
    root, field, model = example
    mutation(model)
    save(root, model)
    with pytest.raises(ConceptGraphError, match=error):
        load_taxonomy(root, field)


def test_nonlocal_cycle_rejected_even_with_an_independent_root(example):
    root, field, model = example
    model["topics"].append({"id": "mlsys.topic.other", "label": "Other", "definition": "A third topic.", "status": "draft"})
    for narrower, broader in [("compression", "representation"), ("representation", "other"), ("other", "compression")]:
        model["hierarchy"].append({"narrower": "mlsys.topic." + narrower, "broader": "mlsys.topic." + broader, "rationale": "An asserted classification.", "status": "draft"})
    save(root, model)
    with pytest.raises(ConceptGraphError, match="hierarchy cycle"):
        load_taxonomy(root, field)


def test_skos_exports_authored_broader_and_collections_without_technical_edges(example):
    root, field, model = example
    field["concepts"][0]["definition"] = 'A "quoted" definition.\nSecond line.'
    save(root, model)
    taxonomy = load_taxonomy(root, field)
    artifacts = taxonomy_artifacts(root, field, taxonomy)
    ttl = artifacts[root / "site/explore/taxonomy.ttl"].decode()
    md = artifacts[root / "site/explore/taxonomy-reference.md"].decode()
    assert ttl.count("a skos:Concept ;") == 4
    assert ttl.count(" skos:broader ") == 2
    assert " skos:narrower " not in ttl  # Standard inverse semantics need no duplicate assertions.
    assert " skos:related " not in ttl  # Technical relationships have distinct semantics.
    assert 'skos:altLabel "Reduced precision representation"@en' in ttl
    assert 'A \\"quoted\\" definition.\\nSecond line.' in ttl
    assert "a skos:Collection" in ttl and ttl.count(" skos:member ") == 2
    assert "dcterms:source <https://www.w3.org/TR/skos-reference/>" in ttl
    assert "Quantization restricts value representation." in md
    assert "collections" in md.lower() and "provisional" in md
    assert "mlsys.memory" in md
    assert taxonomy_artifacts(root, field, taxonomy) == artifacts


def test_collection_membership_resolves_from_one_source_of_truth(example):
    root, field, model = example
    field["concepts"].append({"id": "mlsys.new", "label": "New resource concept", "definition": "A newly authored field concept.", "family": "resources"})
    save(root, model)
    taxonomy = load_taxonomy(root, field)
    assert taxonomy["collections"][0]["members"] == [n["id"] for n in field["concepts"]]
    assert "members" not in model["collections"][0]
    assert taxonomy["roots"] == model["top_concepts"]
    assert not any(e["narrower"] == "mlsys.new" for e in taxonomy["hierarchy"])


def test_collection_cannot_repeat_a_family_under_different_identity(example):
    root, field, model = example
    second = {**model["collections"][0], "id": "mlsys.collection.other"}
    model["collections"].append(second)
    save(root, model)
    with pytest.raises(ConceptGraphError, match="duplicate collection family"):
        load_taxonomy(root, field)


def test_source_cannot_break_turtle_iri_syntax(example):
    root, field, model = example
    model["sources"][0]["url"] = "https://example.org/> . <https://example.org/evil>"
    save(root, model)
    with pytest.raises(ConceptGraphError, match="unsafe source URL"):
        load_taxonomy(root, field)
