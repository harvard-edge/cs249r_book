"""A candidate framework never invents classification or certified coverage."""
from copy import deepcopy
import csv
import io
import json

import pytest
import yaml

from core.concept_graph import ConceptGraphError
from core.framework import framework_artifacts, load_framework


@pytest.fixture
def candidate(tmp_path):
    model = dict(schema_version=1, id="mlsys.framework", title="ML systems candidate framework", version="0.1-draft",
                 status="editorial-draft", scope="A provisional subject boundary.",
                 sources=[dict(id="skos", title="SKOS", url="https://www.w3.org/TR/skos-reference/", role="Design reference")],
                 dimensions={name: [dict(id=name + ".one", label=name.title())] for name in ("activities", "layers", "concerns")},
                 areas=[dict(id="area.execution", label="Execution area", question="How does work execute?", scope="Runtime concerns.",
                             rows=[dict(id="row.state", label="Retained state", definition="The resources retained by execution.",
                                        concepts=["mlsys.state", "mlsys.capacity"], activities=["activities.one"], layers=["layers.one"], concerns=["concerns.one"])])],
                 gaps=[dict(id="gap.candidate", label="Candidate topic", reason="Investigate the boundary.", status="candidate")])
    field = dict(concepts=[dict(id="mlsys.state", label="State", definition="Retained information."),
                           dict(id="mlsys.capacity", label="Capacity", definition="Available storage.")],
                 relationships=[dict(id="relation", source="mlsys.state", target="mlsys.capacity", predicate="occupies")],
                 book_links=[dict(concept="mlsys.state", book_concept="vol1.state", volume="vol1", label="State", chapter_title="Execution", url="https://mlsysbook.ai/state", depth="not-assessed", status="draft")])
    path = tmp_path / "shared/concepts/ml-systems-framework.yml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(model, sort_keys=False))
    return tmp_path, model, field


def write(root, model):
    (root / "shared/concepts/ml-systems-framework.yml").write_text(yaml.safe_dump(model, sort_keys=False))


def test_missing_optional_framework_preserves_old_build(tmp_path):
    assert load_framework(tmp_path, {}) is None
    assert framework_artifacts(tmp_path, {}, None) == {}


def test_candidate_home_links_and_empty_contexts_do_not_claim_coverage(candidate):
    root, original, field = candidate
    before = deepcopy(field)
    framework = load_framework(root, field)
    row = framework["areas"][0]["rows"][0]
    assert row["concept_count"] == 2
    assert row["linked_concept_count"] == 1
    assert row["book_link_count"] == 1
    assert row["book_volumes"] == ["vol1"]
    assert row["contexts"] == []
    assert framework["summary"] == dict(groups=0, areas=1, rows=1, concepts=2, book_links=1, linked_concepts=1, gaps=1)
    assert "coverage" not in row and "competence" not in row
    assert field == before
    assert original["areas"][0]["rows"][0].get("contexts") is None


@pytest.mark.parametrize("mutation,error", [
    (lambda m: m.update(status="reviewed"), "editorial-draft"),
    (lambda m: m.update(version="1.0"), "0.1-draft"),
    (lambda m: m["sources"][0].update(url="javascript:alert(1)"), "unsafe source"),
    (lambda m: m["sources"].append(deepcopy(m["sources"][0])), "duplicate source ID"),
    (lambda m: m["dimensions"]["activities"].append(dict(id="activities.two", label="Activities")), "duplicate activities label"),
    (lambda m: m["areas"][0]["rows"][0].update(activities=["nonexistent"]), "unknown.*activities"),
    (lambda m: m["areas"][0]["rows"][0].update(concepts=["mlsys.unknown"]), "unknown.*concepts"),
    (lambda m: m["areas"][0]["rows"][0].update(concepts=["mlsys.state"]), "unassigned.*capacity"),
    (lambda m: m["areas"][0]["rows"][0].update(concepts=["mlsys.state", "mlsys.state", "mlsys.capacity"]), "duplicate.*concepts"),
    (lambda m: m["areas"][0]["rows"][0].update(status="reviewed"), "remain draft"),
    (lambda m: m["gaps"][0].update(status="confirmed"), "candidate status"),
    (lambda m: m["areas"][0]["rows"][0].update(id="mlsys.state"), "overlapping row identity"),
    (lambda m: m["areas"][0].update(group="stack"), "no groups are defined"),
    (lambda m: m.update(groups=[dict(id="stack", label="Stack", question="What runs it?")]), "requires a known group"),
    (lambda m: (m.update(groups=[dict(id="stack", label="Stack", question="What runs it?"), dict(id="spare", label="Spare", question="Unused?")]),
                m["areas"][0].update(group="stack")), "groups without areas: spare"),
    (lambda m: (m.update(groups=[dict(id="row.state", label="Stack", question="What runs it?")]),
                m["areas"][0].update(group="row.state")), "overlapping row identity"),
    (lambda m: m["areas"][0]["rows"][0].update(book_chapters=["chapter five"]), "malformed chapter code"),
    (lambda m: m["areas"][0]["rows"][0].update(book_chapters=["v1/07"]), "does not resolve"),
    (lambda m: m["areas"][0]["rows"][0].update(excludes=["Storage ([[row.missing]])"]), "cites unknown row or area row.missing"),
    (lambda m: m["sources"][0].update(**{"run rules).": None}), "unknown keys"),
])
def test_bad_edits_fail_before_exports(candidate, mutation, error):
    root, model, field = candidate
    mutation(model)
    write(root, model)
    with pytest.raises(ConceptGraphError, match=error):
        load_framework(root, field)


def test_duplicate_home_across_distinct_rows_rejected(candidate):
    root, model, field = candidate
    other = deepcopy(model["areas"][0]["rows"][0])
    other.update(id="row.other", label="Other row")
    model["areas"][0]["rows"].append(other)
    write(root, model)
    with pytest.raises(ConceptGraphError, match="exactly one candidate home"):
        load_framework(root, field)


def test_new_field_concept_requires_explicit_home_decision(candidate):
    root, _, field = candidate
    field["concepts"].append(dict(id="mlsys.new", label="New concept", definition="A new identity."))
    with pytest.raises(ConceptGraphError, match="unassigned.*mlsys.new"):
        load_framework(root, field)


def test_contexts_are_optional_but_must_be_known_when_authored(candidate):
    root, model, field = candidate
    model["dimensions"]["contexts"] = [dict(id="context.physical", label="Physical systems")]
    model["areas"][0]["rows"][0]["contexts"] = ["context.physical"]
    write(root, model)
    assert load_framework(root, field)["areas"][0]["rows"][0]["contexts"] == ["context.physical"]
    model["areas"][0]["rows"][0]["contexts"] = ["context.unknown"]
    write(root, model)
    with pytest.raises(ConceptGraphError, match="unknown.*contexts"):
        load_framework(root, field)


def test_exports_share_authored_rows_and_preserve_unmapped_concepts(candidate):
    root, model, field = candidate
    model["areas"][0]["rows"][0]["definition"] = 'Resources, including "state".\nA second line.'
    write(root, model)
    framework = load_framework(root, field)
    artifacts = framework_artifacts(root, field, framework)
    site = root / "site/taxonomy"
    # Three internal files plus six public downloads.
    assert len(artifacts) == 9
    data = json.loads(artifacts[site / "framework-data.json"])
    assert data["concepts"] == field["concepts"]
    assert data["relationships"] == field["relationships"]
    assert data["book_links"] == field["book_links"]
    rows = list(csv.DictReader(io.StringIO(artifacts[site / "framework-table.csv"].decode())))
    assert len(rows) == 1
    assert rows[0]["definition"] == model["areas"][0]["rows"][0]["definition"]
    assert rows[0]["concept_ids"] == "mlsys.state;mlsys.capacity"
    assert rows[0]["linked_concept_count"] == "1"
    reference = artifacts[site / "framework-reference.md"].decode()
    assert "candidate editorial synthesis" in reference
    assert "teaching depth not assessed" in reference
    assert "Capacity" in reference
    assert artifacts == framework_artifacts(root, field, framework)


def test_empty_candidate_row_and_secondary_links_do_not_inflate_home_counts(candidate):
    root, model, field = candidate
    model["areas"][0]["rows"].append(dict(id="row.proposed", label="Proposed subject", definition="A candidate beyond the current inventory.",
        concepts=[], secondary_concepts=["mlsys.state"], activities=["activities.one"], layers=["layers.one"], concerns=["concerns.one"],
        includes=["Retained state implications"], excludes=["Storage ownership"], warrants=["skos"]))
    write(root, model)
    framework = load_framework(root, field)
    row = framework["areas"][0]["rows"][1]
    assert row["concept_count"] == row["home_concept_count"] == 0
    assert row["secondary_concept_count"] == 1
    assert row["book_link_count"] == 1
    assert framework["summary"]["concepts"] == 2
    assert framework["summary"]["book_links"] == 1
    artifacts = framework_artifacts(root, field, framework)
    site = root / "site/taxonomy"
    rows = list(csv.DictReader(io.StringIO(artifacts[site / "framework-table.csv"].decode())))
    assert len(rows) == 2
    assert rows[1]["concept_ids"] == ""
    assert rows[1]["secondary_concept_ids"] == "mlsys.state"
    assert rows[1]["secondary_concept_count"] == "1"
    reference = artifacts[site / "framework-reference.md"].decode()
    assert "proposed subject scope, not evidence of absence" in reference
    assert "Secondary concept connections" in reference
    assert "Retained state implications" in reference


@pytest.mark.parametrize("edit,error", [
    (dict(secondary_concepts=["mlsys.state"]), "overlap own home"),
    (dict(secondary_concepts=["unknown"]), "unknown.*secondary"),
    (dict(secondary_concepts=["mlsys.state", "mlsys.state"]), "duplicate.*secondary"),
    (dict(includes=[]), "includes.*nonempty text"),
    (dict(excludes=[""]), "excludes.*nonempty text"),
    (dict(warrants=["unknown"]), "unknown.*warrants"),
    (dict(label="Activities"), "duplicates a facet label"),
])
def test_secondary_scope_and_mixed_axis_edits_fail(candidate, edit, error):
    root, model, field = candidate
    model["areas"][0]["rows"][0].update(edit)
    write(root, model)
    with pytest.raises(ConceptGraphError, match=error):
        load_framework(root, field)


def test_csv_preserves_scope_boundaries_and_source_warrants(candidate):
    root, model, field = candidate
    area = model["areas"][0]
    area["display_label"] = "Execution"
    row = area["rows"][0]
    row.update(display_label="State", includes=["State; and capacity", "Working memory"], excludes=["External archives"],
               warrants=["skos"], warrant_role="design-reading-not-claim-proof")
    write(root, model)
    framework = load_framework(root, field)
    exports = framework_artifacts(root, field, framework)
    result = list(csv.DictReader(io.StringIO(exports[root / "site/taxonomy/framework-table.csv"].decode())))[0]
    assert json.loads(result["includes"]) == row["includes"]
    assert json.loads(result["excludes"]) == row["excludes"]
    assert result["warrant_source_ids"] == "skos"
    assert json.loads(result["warrant_source_urls"]) == [model["sources"][0]["url"]]
    assert json.loads(result["warrant_source_titles"]) == ["SKOS"]
    assert result["warrant_role"] == "design-reading-not-claim-proof"
    assert result["inventory_status"] == "draft-inventory"
    assert result["source_file"] == "shared/concepts/ml-systems-framework.yml"
    assert result["area_question"] == area["question"]
    assert result["row_display_label"] == "State"


def test_gap_label_derives_from_subject_row_without_duplicate_text(candidate):
    root, model, field = candidate
    gap = model["gaps"][0]
    gap.update(subject_row="row.state")
    del gap["label"]
    write(root, model)
    assert load_framework(root, field)["gaps"][0]["label"] == "Retained state"
    gap["label"] = "Outdated duplicated label"
    write(root, model)
    assert load_framework(root, field)["gaps"][0]["label"] == "Retained state"
    gap["subject_row"] = "row.unknown"
    write(root, model)
    with pytest.raises(ConceptGraphError, match="unknown gap subject_row"):
        load_framework(root, field)


@pytest.mark.parametrize("where,value", [("area", ""), ("row", "  "), ("row", ["State"])])
def test_display_labels_require_nonempty_strings(candidate, where, value):
    root, model, field = candidate
    target = model["areas"][0] if where == "area" else model["areas"][0]["rows"][0]
    target["display_label"] = value
    write(root, model)
    with pytest.raises(ConceptGraphError, match="requires display_label"):
        load_framework(root, field)


def test_groups_and_chapter_pointers_resolve_without_claiming_depth(candidate):
    root, model, field = candidate
    chapter = root / "books/vol2/08_fleet_orchestration"
    chapter.mkdir(parents=True)
    (chapter / "08_fleet_orchestration.qmd").write_text("# Fleet Orchestration {#sec-fleet}\n\nBody.\n")
    model.update(version="0.2-draft", groups=[dict(id="stack", label="Stack", question="What does it run on?")])
    model["areas"][0].update(group="stack")
    model["areas"][0]["rows"][0].update(book_chapters=["v2/08"])
    write(root, model)
    framework = load_framework(root, field)
    assert framework["summary"]["groups"] == 1
    assert framework["areas"][0]["rows"][0]["book_chapters"] == [dict(
        code="vol2/08", volume="vol2", chapter="vol2/08_fleet_orchestration", title="Fleet Orchestration",
        url="https://mlsysbook.ai/vol2/08_fleet_orchestration/08_fleet_orchestration.html")]
    artifacts = {path.name: data.decode() for path, data in framework_artifacts(root, field, framework).items()}
    assert "## Stack" in artifacts["framework-reference.md"]
    assert "### Execution area" in artifacts["framework-reference.md"]
    rows = list(csv.DictReader(io.StringIO(artifacts["framework-table.csv"])))
    assert rows[0]["group_id"] == "stack" and rows[0]["group_label"] == "Stack"


def test_chapter_pointer_ignores_leftover_build_folders(candidate):
    # 2026-10-27: an untracked books/vol1/02_ml_systems/ (old rendered HTML) beside the renamed
    # 02_system_paradigms/ made `v1/02` ambiguous and failed every concepts build in the main checkout.
    root, model, field = candidate
    chapter = root / "books/vol1/02_system_paradigms"
    chapter.mkdir(parents=True)
    (chapter / "02_system_paradigms.qmd").write_text("# System Paradigms\n")
    leftover = root / "books/vol1/02_ml_systems"
    leftover.mkdir(parents=True)
    (leftover / "02_ml_systems.html").write_text("<html></html>")
    model["areas"][0]["rows"][0].update(book_chapters=["v1/02"])
    write(root, model)
    framework = load_framework(root, field)
    assert framework["areas"][0]["rows"][0]["book_chapters"][0]["chapter"] == "vol1/02_system_paradigms"


def test_scope_references_render_current_labels(candidate):
    root, model, field = candidate
    model["areas"][0]["rows"][0].update(excludes=["Persisting state ([[row.state]]) and the area ([[area.execution]])"])
    write(root, model)
    framework = load_framework(root, field)
    reference = {path.name: data.decode() for path, data in framework_artifacts(root, field, framework).items()}["framework-reference.md"]
    assert "Persisting state (Retained state) and the area (Execution area)" in reference
    assert "[[" not in reference


def _coded(model):
    model.update(version="0.2-draft", groups=[dict(id="stack", code="S", label="Stack", question="What does it run on?")])
    model["areas"][0].update(group="stack", code="S1")
    model["areas"][0]["rows"][0].update(code="S1.1")
    return model


def test_readable_codes_must_nest_and_reach_every_download(candidate):
    root, model, field = candidate
    write(root, _coded(model))
    framework = load_framework(root, field)
    artifacts = {path.name: data.decode() for path, data in framework_artifacts(root, field, framework).items()}
    assert {"mlsys-taxonomy.json", "mlsys-taxonomy-subjects.csv", "mlsys-taxonomy-concepts.csv", "mlsys-taxonomy.ttl",
            "mlsys-concept-graph.graphml", "mlsys-taxonomy.bib"} <= set(artifacts)
    outline = json.loads(artifacts["mlsys-taxonomy.json"])
    assert outline["groups"][0]["areas"][0]["subjects"][0]["code"] == "S1.1"
    assert {c["home"] for c in outline["concepts"]} == {"S1.1"}
    assert artifacts["mlsys-taxonomy-subjects.csv"].splitlines()[1].startswith("S1.1,Stack,")
    assert 'skos:notation "S1.1"' in artifacts["mlsys-taxonomy.ttl"]
    assert artifacts["mlsys-concept-graph.graphml"].count("<node ") == 2
    # Researchers and instructors read gaps and book passages from the release, not the YAML.
    assert outline["gaps"] == [{"id": "gap.candidate", "topic": "", "subject": "", "status": "candidate",
                                "reason": "Investigate the boundary."}]
    state = next(c for c in outline["concepts"] if c["id"] == "mlsys.state")
    assert state["book_passages"] == [{"volume": "vol1", "chapter": "Execution", "url": "https://mlsysbook.ai/state"}]
    assert artifacts["mlsys-taxonomy-concepts.csv"].splitlines()[0].endswith(",background,taught_in")


@pytest.mark.parametrize("edit,error", [
    (lambda m: m["areas"][0]["rows"][0].update(code="S2.1"), "does not extend area code"),
    (lambda m: m["areas"][0].update(code="L1"), "does not extend group code"),
    (lambda m: m["areas"][0]["rows"][0].pop("code"), "requires a unique readable code"),
])
def test_bad_codes_fail(candidate, edit, error):
    root, model, field = candidate
    edit(_coded(model))
    write(root, model)
    with pytest.raises(ConceptGraphError, match=error):
        load_framework(root, field)


def test_v03_enforces_two_relationships_per_concept_and_connected_graph(candidate):
    root, model, field = candidate
    model["version"] = "0.3-draft"
    write(root, model)
    with pytest.raises(ConceptGraphError, match="at least two relationships"):
        load_framework(root, field)
    # Two separate 3-cycles satisfy degree >= 2 everywhere but are disconnected.
    ids = [f"mlsys.c{i}" for i in range(6)]
    field["concepts"] = [dict(id=c, label=c, definition="A concept.") for c in ids]
    field["relationships"] = [
        dict(id=f"r{i}", source=a, target=b, predicate="links")
        for i, (a, b) in enumerate([
            ("mlsys.c0", "mlsys.c1"), ("mlsys.c1", "mlsys.c2"), ("mlsys.c2", "mlsys.c0"),
            ("mlsys.c3", "mlsys.c4"), ("mlsys.c4", "mlsys.c5"), ("mlsys.c5", "mlsys.c3"),
        ])
    ]
    model["areas"][0]["rows"][0]["concepts"] = ids
    write(root, model)
    with pytest.raises(ConceptGraphError, match="disconnected"):
        load_framework(root, field)
    field["relationships"].append(dict(id="bridge", source="mlsys.c2", target="mlsys.c3", predicate="links"))
    assert load_framework(root, field)["summary"]["concepts"] == 6
