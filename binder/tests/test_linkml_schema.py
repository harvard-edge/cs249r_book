"""LinkML metamodel and ontological graph validation tests for the ML Systems Taxonomy."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import jsonschema
import pytest
import yaml

from core.concept_graph import ConceptGraphError
from core.field_graph import load_taxonomy_yaml
from core.taxonomy_schema import (
    compile_json_schema,
    get_schema_view,
    resolve_schema_path,
    validate_field_semantics,
    validate_guides_semantics,
    validate_linkml,
    validate_schema_sync,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_linkml_schema_compiles_to_valid_json_schema():
    schema_path = resolve_schema_path(REPO_ROOT)
    assert schema_path.exists()
    sv = get_schema_view(REPO_ROOT)
    assert sv is not None
    compiled = compile_json_schema(REPO_ROOT)
    assert "$defs" in compiled
    for cls_name in (
        "FieldModel",
        "FrameworkModel",
        "GuidesModel",
        "BookAlignmentModel",
        "TaxonomyHierarchyModel",
        "CoverageChecklistModel",
        "CoverageLedgerModel",
        "AgreementRecordModel",
        "AgreementItemTag",
        "TaxonomyReleaseDocument",
    ):
        assert cls_name in compiled["$defs"]


@pytest.mark.parametrize(
    "rel_path,target_class",
    [
        ("taxonomy/ml-systems-field.yml", "FieldModel"),
        ("taxonomy/ml-systems-framework.yml", "FrameworkModel"),
        ("taxonomy/ml-systems-guides.yml", "GuidesModel"),
        ("taxonomy/mlsysbook-alignment.yml", "BookAlignmentModel"),
        ("taxonomy/ml-systems-taxonomy.yml", "TaxonomyHierarchyModel"),
        ("taxonomy/coverage-ledger.yml", "CoverageLedgerModel"),
        ("taxonomy/coverage/courses-and-standards-2026.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/mlsys-papers-2025-2026.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/mlsysbook-vol1-vol2-primary.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/mlsysbook-vol3-vol4-primary.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/systems-venues-2025.yml", "CoverageChecklistModel"),
        ("taxonomy/agreement/staffml-300-2026-10.yml", "AgreementRecordModel"),
        ("taxonomy/agreement/staffml-300-heldout-2026-10.yml", "AgreementRecordModel"),
    ],
)
def test_authored_taxonomy_yamls_conform_to_linkml_metamodel(rel_path: str, target_class: str):
    path = REPO_ROOT / rel_path
    instance = load_taxonomy_yaml(path)
    if target_class == "CoverageLedgerModel":
        instance = {
            **instance,
            "decisions": [
                {**d, "reviewed_on": str(d["reviewed_on"])} for d in instance.get("decisions", [])
            ],
        }
    validate_linkml(instance, target_class, REPO_ROOT, rel_path)
    if target_class == "AgreementRecordModel":
        for run in instance.get("runs", []):
            for tagger, tags in (run.get("taggers") or {}).items():
                for t in tags:
                    validate_linkml(t, "AgreementItemTag", REPO_ROOT, f"{rel_path}:{tagger}")


def test_linkml_rejects_malformed_concept_id_and_unknown_slot():
    field = load_taxonomy_yaml(REPO_ROOT / "taxonomy/ml-systems-field.yml")
    bad = deepcopy(field)
    bad["concepts"][0]["id"] = "Invalid_ID"
    with pytest.raises(ConceptGraphError, match="LinkML FieldModel validation failed"):
        validate_linkml(bad, "FieldModel", REPO_ROOT, "field model")

    bad_slot = deepcopy(field)
    bad_slot["unexpected_top_slot"] = "not in LinkML"
    with pytest.raises(ConceptGraphError, match="LinkML FieldModel validation failed"):
        validate_linkml(bad_slot, "FieldModel", REPO_ROOT, "field model")


@pytest.mark.parametrize("rel_name", ["specializes", "part-of", "requires"])
def test_ontological_dag_check_rejects_hierarchical_cycles(rel_name: str):
    model = {
        "relationships": [
            {"id": "r1", "source": "mlsys.a", "target": "mlsys.b", "relation": rel_name, "predicate": "links"},
            {"id": "r2", "source": "mlsys.b", "target": "mlsys.c", "relation": rel_name, "predicate": "links"},
            {"id": "r3", "source": "mlsys.c", "target": "mlsys.a", "relation": rel_name, "predicate": "links"},
        ]
    }
    with pytest.raises(ConceptGraphError, match=f"ontological cycle in '{rel_name}'"):
        validate_field_semantics(model)


@pytest.mark.parametrize(
    "relation,predicate",
    [
        ("increases", "shrinks the safe stopping envelope"),
        ("increases", "reduces memory bandwidth traffic"),
        ("reduces", "increases tail latency under load"),
        ("mitigates", "amplifies synchronization stalls"),
    ],
)
def test_polarity_check_rejects_contradictory_predicates(relation: str, predicate: str):
    model = {
        "relationships": [
            {
                "id": "rel.bad.polarity",
                "source": "mlsys.a",
                "target": "mlsys.b",
                "relation": relation,
                "predicate": predicate,
            }
        ]
    }
    with pytest.raises(ConceptGraphError, match="contradicts relation"):
        validate_field_semantics(model)


def test_guides_semantics_rejects_overlapping_symptom_buckets_and_unassigned_subjects():
    guides = {
        "roles": [{"id": "r1", "core": [f"s{i}" for i in range(19)], "working": []}],
        "symptoms": [
            {
                "id": "sym-overlap",
                "label": "Overlap",
                "signal": "Signal",
                "causes": ["mlsys.a"],
                "checks": ["mlsys.b"],
                "levers": ["mlsys.a"],
            }
        ],
        "learning_paths": [],
    }
    rows_by_id = {f"s{i}": {} for i in range(20)}
    with pytest.raises(ConceptGraphError, match="both cause and lever: mlsys.a"):
        validate_guides_semantics(guides, rows_by_id, version="0.3-draft")

    guides["symptoms"][0]["levers"] = ["mlsys.c"]
    with pytest.raises(ConceptGraphError, match="subjects missing from all role guides: s19"):
        validate_guides_semantics(guides, rows_by_id, version="0.3-draft")


def test_staffml_taxonomy_json_stays_synced_with_compiled_release():
    staffml_json = REPO_ROOT / "staffml/app/src/data/mlsys-taxonomy.json"
    release_json = REPO_ROOT / "taxonomy/downloads/mlsys-taxonomy.json"
    if staffml_json.exists() and release_json.exists():
        assert json.loads(staffml_json.read_text()) == json.loads(release_json.read_text())


def test_schema_enums_stay_synced_with_compiler_and_field_relations():
    field = load_taxonomy_yaml(REPO_ROOT / "taxonomy/ml-systems-field.yml")
    validate_schema_sync(REPO_ROOT, field)
    bad = deepcopy(field)
    bad["relations"].append(
        {"id": "invented-verb", "forward": "invents", "inverse": "invented by", "kind": "mechanism", "meaning": "X"}
    )
    with pytest.raises(ConceptGraphError, match="RelationIdEnum disagrees"):
        validate_schema_sync(REPO_ROOT, bad)


def test_public_release_downloads_validate_end_to_end():
    downloads = REPO_ROOT / "taxonomy/downloads"
    if not (downloads / "mlsys-taxonomy.json").exists():
        pytest.skip("downloads not yet built")

    schema = json.loads((downloads / "mlsys-taxonomy.schema.json").read_text())
    release = json.loads((downloads / "mlsys-taxonomy.json").read_text())

    # 1. Public JSON release validates directly against public JSON Schema.
    jsonschema.validate(release, schema)

    # Corrupting a relation verb or concept ID in the release fails JSON Schema validation.
    bad_rel = deepcopy(release)
    bad_rel["relationships"][0]["relation"] = "not-a-valid-relation"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_rel, schema)

    bad_cid = deepcopy(release)
    bad_cid["relationships"][0]["source"] = "bad_prefix.kv-cache"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_cid, schema)

    # 2. GraphML parses cleanly as XML with exact node and edge counts matching the JSON release.
    tree = ET.parse(downloads / "mlsys-concept-graph.graphml")
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    nodes = tree.findall(".//g:node", ns)
    edges = tree.findall(".//g:edge", ns)
    assert len(nodes) == len(release["concepts"])
    assert len(edges) == len(release["relationships"])
    node_ids = {n.attrib["id"] for n in nodes}
    assert all(e.attrib["source"] in node_ids and e.attrib["target"] in node_ids for e in edges)

    # 3. Subjects CSV and Concepts CSV match the compiled JSON release.
    subj_rows = list(csv.DictReader(io.StringIO((downloads / "mlsys-taxonomy-subjects.csv").read_text())))
    conc_rows = list(csv.DictReader(io.StringIO((downloads / "mlsys-taxonomy-concepts.csv").read_text())))
    expected_subjects = sum(len(a["subjects"]) for g in release["groups"] for a in g["areas"])
    assert len(subj_rows) == expected_subjects
    assert len(conc_rows) == len(release["concepts"])

    # 4. SKOS Turtle (.ttl) contains all prefixes, 17 object properties, collections, and concepts.
    ttl_text = (downloads / "mlsys-taxonomy.ttl").read_text()
    assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#> ." in ttl_text
    assert ttl_text.count("a owl:ObjectProperty ;") == len(release["relations"])
    assert ttl_text.count("a skos:Concept ;") == len(release["concepts"])


