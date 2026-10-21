"""Field identity must survive book changes without inventing coverage or edges."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from core.concept_graph import ConceptGraphError
from core.field_graph import field_artifacts, load_field


@pytest.fixture
def landscape(tmp_path):
    folder = tmp_path / "shared/concepts"
    folder.mkdir(parents=True)
    model = dict(
        schema_version=1, id="example", title="Field question?", scope="A declared field boundary.", status="editorial-draft",
        sources=[dict(id="primary", title="Primary reading", url="https://example.org/source", kind="research", scope="A reading basis.")],
        families=[dict(id="systems", label="Systems", scope="core", question="What limits work?", references=["primary"])],
        concepts=[dict(id="mlsys.capacity", label="Capacity", definition="Available storage.", family="systems", status="draft"),
                  dict(id="mlsys.state", label="State", definition="Retained information.", family="systems", status="draft"),
                  dict(id="mlsys.unmapped", label="Unmapped concept", definition="An independently selected concept.", family="systems", status="draft")],
        relationships=[dict(id="rel.state.capacity", source="mlsys.state", target="mlsys.capacity", predicate="occupies", conditions="For resident state.", basis="editorial-synthesis", kind="constraint", status="draft", references=["primary"])],
        views=[dict(id="overview", title="Selected question?", concepts=["mlsys.capacity", "mlsys.state"]),
               dict(id="all", title="Whole field?", concepts=["mlsys.capacity", "mlsys.state", "mlsys.unmapped"])],
    )
    (folder / "ml-systems-field.yml").write_text(yaml.safe_dump(model))
    return tmp_path, model


def save(root, model):
    (root / "shared/concepts/ml-systems-field.yml").write_text(yaml.safe_dump(model))


def test_field_identity_and_unmapped_concepts_survive_empty_books(landscape):
    root, _ = landscape
    field = load_field(root, {"nodes": {}, "chapters": {}})
    assert field["summary"]["concepts"] == 3
    assert field["summary"]["linked_concepts"] == 0
    assert field["summary"]["unconnected_concepts"] == ["mlsys.unmapped"]
    assert len(field["relationships"]) == 1  # No invented edge to eliminate an orphan.
    assert not any("owner" in c or "chapter" in c for c in field["concepts"])


@pytest.mark.parametrize("mutation, error", [
    (lambda m: m["concepts"].append(deepcopy(m["concepts"][0])), "duplicate concept"),
    (lambda m: m["relationships"][0].update(target="mlsys.missing"), "endpoints"),
    (lambda m: m["relationships"][0].update(conditions=""), "requires conditions"),
    (lambda m: m["relationships"][0].update(references=["unrecorded"]), "reading references"),
    (lambda m: m["relationships"][0].update(status="reviewed"), "masquerade"),
    (lambda m: m["concepts"][0].update(status="reviewed"), "review label"),
    (lambda m: m["sources"][0].update(url="javascript:alert(1)"), "unsafe reading URL"),
    (lambda m: m["views"][1]["concepts"].pop(), "retain every"),
    (lambda m: m["relationships"][0].update(kind="prerequisite"), "relationship kind"),
])
def test_invalid_model_cannot_publish_false_identity_or_certainty(landscape, mutation, error):
    root, model = landscape
    mutation(model)
    save(root, model)
    with pytest.raises(ConceptGraphError, match=error):
        load_field(root, {"nodes": {}, "chapters": {}})


def test_scope_is_independent_of_navigation_family(landscape):
    root, model = landscape
    model["concepts"][1]["scope"] = "generative-extension"
    save(root, model)
    field = load_field(root, {"nodes": {}, "chapters": {}})
    assert field["concepts"][1]["family"] == "systems"
    assert field["concepts"][1]["scope"] == "generative-extension"


def test_book_links_are_explicit_candidates_not_coverage(landscape):
    root, _ = landscape
    folder = root / "books/vol1/01_intro"
    folder.mkdir(parents=True)
    source = folder / "intro.qmd"
    source.write_text("# State {#sec-state}\nResident state uses capacity.\n")
    book = {"nodes": {"vol1.state": dict(id="vol1.state", label="State", chapter="vol1/01_intro", volume=1, section="sec-state", url="https://mlsysbook.ai/state#sec-state")},
            "chapters": {"vol1/01_intro": dict(source=source, title="State", sections={"sec-state": source.read_text()})}}
    link = dict(concept="mlsys.state", book_concept="vol1.state", status="draft", depth="not-assessed")
    path = root / "shared/concepts/mlsysbook-alignment.yml"
    path.write_text(yaml.safe_dump(dict(schema_version=1, links=[link])))
    field = load_field(root, book)
    assert field["summary"]["linked_concepts"] == 1
    assert field["book_links"][0]["depth"] == "not-assessed"
    assert "source_hash" in field["book_links"][0]
    # Neither identical words nor an existing related node creates another link.
    assert len(field["book_links"]) == 1
    link["depth"] = "adequate"
    path.write_text(yaml.safe_dump(dict(schema_version=1, links=[link])))
    with pytest.raises(ConceptGraphError, match="certify coverage"):
        load_field(root, book)
    link.update(depth="not-assessed", relation="same-concept")
    path.write_text(yaml.safe_dump(dict(schema_version=1, links=[link])))
    with pytest.raises(ConceptGraphError, match="conceptual equivalence"):
        load_field(root, book)


def test_deleted_book_identity_is_not_silently_reassigned(landscape):
    root, _ = landscape
    path = root / "shared/concepts/mlsysbook-alignment.yml"
    path.write_text(yaml.safe_dump(dict(schema_version=1, links=[dict(concept="mlsys.state", book_concept="vol1.deleted", status="draft", depth="not-assessed")])))
    with pytest.raises(ConceptGraphError, match="unresolved book link"):
        load_field(root, {"nodes": {}, "chapters": {}})


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz required")
def test_portable_view_and_svg_use_same_records_without_script_injection(landscape):
    root, model = landscape
    model["concepts"][0]["definition"] = "</script><script>window.bad=true</script>"
    save(root, model)
    site = root / "site/explore"
    site.mkdir(parents=True)
    (site / "field.qmd").write_text('---\n---\n```{=html}\n<div>Field</div>\n```\n')
    artifacts = field_artifacts(root, {"nodes": {}, "chapters": {}})
    data = json.loads(artifacts[site / "field-data.json"])
    assert data["summary"]["concepts"] == 3
    assert "mlsys.unmapped" in data["diagrams"]["all"]
    assert "mlsys.unmapped" not in data["diagrams"]["overview"]
    assert set(data["neighborhoods"]["mlsys.state"]) == {"mlsys.state", "mlsys.capacity"}
    assert data["neighborhoods"]["mlsys.unmapped"] == ["mlsys.unmapped"]
    assert "mlsys.unmapped" not in data["diagrams"]["focus-mlsys.state"]
    assert site / "field-focus/mlsys.state.svg" in artifacts
    assert "<script>window.bad" not in artifacts[site / "field-preview.html"].decode()
    assert artifacts[site / "field-all.svg"].startswith(b"<?xml")
