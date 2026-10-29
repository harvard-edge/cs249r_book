"""LinkML metamodel and ontological graph validation tests for the ML Systems Taxonomy."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

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
        ("taxonomy/coverage/mlsysbook-vol1-v2-primary.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/mlsysbook-vol3-vol4-primary.yml", "CoverageChecklistModel"),
        ("taxonomy/coverage/systems-venues-2025.yml", "CoverageChecklistModel"),
    ],
)
def test_authored_taxonomy_yamls_conform_to_linkml_metamodel(rel_path: str, target_class: str):
    path = REPO_ROOT / rel_path
    if not path.exists():
        path = REPO_ROOT / rel_path.replace("vol1-v2", "vol1-vol2")
    instance = load_taxonomy_yaml(path)
    validate_linkml(instance, target_class, REPO_ROOT, rel_path)


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

