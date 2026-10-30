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
from core.field_graph import load_taxonomy_yaml, load_unified_taxonomy
from core.framework import load_framework
from core.taxonomy_release import release_artifacts, staffml_labels_payload
from core.taxonomy_schema import (
    compile_json_schema,
    get_class_slots,
    get_enum_values,
    get_hierarchical_relations,
    get_required_class_slots,
    get_schema_view,
    resolve_schema_path,
    validate_field_semantics,
    validate_guides_semantics,
    validate_linkml,
    validate_schema_sync,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

AUTHORED_MODEL_CASES = [
    ("taxonomy/ml-systems-taxonomy.yml", "UnifiedTaxonomyModel"),
    ("taxonomy/mlsysbook-alignment.yml", "BookAlignmentModel"),
    ("taxonomy/coverage-ledger.yml", "CoverageLedgerModel"),
    *[
        (p.relative_to(REPO_ROOT).as_posix(), "CoverageChecklistModel")
        for p in sorted((REPO_ROOT / "taxonomy/coverage").glob("*.yml"))
    ],
    *[
        (p.relative_to(REPO_ROOT).as_posix(), "AgreementRecordModel")
        for p in sorted((REPO_ROOT / "taxonomy/agreement").glob("*.yml"))
    ],
]


def test_linkml_schema_compiles_to_valid_json_schema():
    schema_path = resolve_schema_path(REPO_ROOT)
    assert schema_path.exists()
    sv = get_schema_view(REPO_ROOT)
    assert sv is not None
    compiled = compile_json_schema(REPO_ROOT)
    assert "$defs" in compiled
    for cls_name in (
        "UnifiedTaxonomyModel",
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
        "StaffMLLabelsDocument",
    ):
        assert cls_name in compiled["$defs"]


@pytest.mark.parametrize("rel_path,target_class", AUTHORED_MODEL_CASES)
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


def test_unified_taxonomy_projections_conform_to_aspect_models(tmp_path: Path):
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    validate_linkml(bundle["field"], "FieldModel", REPO_ROOT, "projected field")
    validate_linkml(bundle["framework"], "FrameworkModel", REPO_ROOT, "projected framework")
    validate_linkml(bundle["guides"], "GuidesModel", REPO_ROOT, "projected guides")
    validate_linkml(bundle["skos"], "TaxonomyHierarchyModel", REPO_ROOT, "projected skos")

    raw = load_taxonomy_yaml(REPO_ROOT / "taxonomy/ml-systems-taxonomy.yml")
    no_home = deepcopy(raw)
    no_home["concepts"][0].pop("home")
    with pytest.raises(ConceptGraphError, match=r"home.*required"):
        validate_linkml(no_home, "UnifiedTaxonomyModel", REPO_ROOT, "taxonomy/ml-systems-taxonomy.yml")

    tax_dir = tmp_path / "taxonomy"
    tax_dir.mkdir(parents=True)
    with_row_concepts = deepcopy(raw)
    with_row_concepts["areas"][0]["rows"][0]["concepts"] = ["mlsys.requirements"]
    (tax_dir / "ml-systems-taxonomy.yml").write_text(yaml.safe_dump(with_row_concepts, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="must not hand-list concepts"):
        load_unified_taxonomy(tmp_path)

    with_all_view = deepcopy(raw)
    with_all_view["views"].append({"id": "all", "title": "All", "concepts": ["mlsys.requirements"]})
    (tax_dir / "ml-systems-taxonomy.yml").write_text(yaml.safe_dump(with_all_view, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="must not hand-list 'all'"):
        load_unified_taxonomy(tmp_path)


def test_linkml_rejects_malformed_concept_id_and_unknown_slot():
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    field = bundle["field"]
    bad = deepcopy(field)
    bad["concepts"][0]["id"] = "Invalid_ID"
    with pytest.raises(ConceptGraphError, match=r"concepts\[0:Invalid_ID\]\.id"):
        validate_linkml(bad, "FieldModel", REPO_ROOT, "field model")

    bad_slot = deepcopy(field)
    bad_slot["unexpected_top_slot"] = "not in LinkML"
    with pytest.raises(ConceptGraphError, match="LinkML FieldModel validation failed"):
        validate_linkml(bad_slot, "FieldModel", REPO_ROOT, "field model")

    bad_agreement = {
        "id": "test-sample",
        "title": "Test sample",
        "basis": "Protocol",
        "items": [{"n": 1}],
        "runs": [
            {
                "id": "r1",
                "held_out": True,
                "taggers": {"t1": [{"n": 0, "subject": "llm-inference", "concepts": ["bad_id"]}]},
            }
        ],
    }
    with pytest.raises(ConceptGraphError, match="LinkML AgreementRecordModel validation failed"):
        validate_linkml(bad_agreement, "AgreementRecordModel", REPO_ROOT, "agreement")
    assert "0.3-draft" in get_enum_values("FrameworkVersionEnum", REPO_ROOT)


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
    assert staffml_json.exists()
    staffml = json.loads(staffml_json.read_text())
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    expected_staffml = staffml_labels_payload(bundle["framework"], bundle["field"])
    validate_linkml(staffml, "StaffMLLabelsDocument", REPO_ROOT, "staffml taxonomy labels")
    assert staffml == expected_staffml
    assert set(staffml["concepts"]) == {c["id"] for c in bundle["field"]["concepts"]}
    assert set(staffml["subjects"]) == {
        r["id"] for a in bundle["framework"]["areas"] for r in a["rows"]
    }


def test_schema_enums_stay_synced_with_compiler_and_field_relations():
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    field = bundle["field"]
    validate_schema_sync(REPO_ROOT, field)
    bad = deepcopy(field)
    bad["relations"].append(
        {"id": "invented-verb", "forward": "invents", "inverse": "invented by", "kind": "mechanism", "meaning": "X"}
    )
    with pytest.raises(ConceptGraphError, match="RelationIdEnum disagrees"):
        validate_schema_sync(REPO_ROOT, bad)

    bad_meaning = deepcopy(field)
    bad_meaning["relations"][0]["meaning"] = "drifted meaning"
    with pytest.raises(ConceptGraphError, match="description disagrees"):
        validate_schema_sync(REPO_ROOT, bad_meaning)

    bad_kind = deepcopy(field)
    bad_kind["relations"][0]["kind"] = "failure"
    with pytest.raises(ConceptGraphError, match="annotation 'kind' disagrees"):
        validate_schema_sync(REPO_ROOT, bad_kind)


def test_unknown_target_class_and_cardinality_and_strict_slot_keys(tmp_path: Path):
    with pytest.raises(ConceptGraphError, match="Unknown LinkML target class: NoSuchClass"):
        compile_json_schema(REPO_ROOT, target_class="NoSuchClass")

    compiled = compile_json_schema(REPO_ROOT)
    rel_prop = compiled["$defs"]["UnifiedTaxonomyModel"]["properties"]["relations"]
    assert rel_prop.get("minItems") == 17
    assert get_hierarchical_relations(REPO_ROOT) == ("specializes", "part-of", "requires")
    assert "home" in get_required_class_slots("UnifiedConcept", REPO_ROOT)
    assert "id" in get_class_slots("SubjectRow", REPO_ROOT)

    schema_dir = tmp_path / "taxonomy/schema"
    schema_dir.mkdir(parents=True)
    raw_schema = yaml.safe_load(resolve_schema_path(REPO_ROOT).read_text())
    bad_schema = deepcopy(raw_schema)
    bad_schema["classes"]["FieldModel"]["attributes"]["id"]["unknown_slot_attr"] = True
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_schema, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="Unsupported LinkML slot keys"):
        compile_json_schema(tmp_path)

    bad_range = deepcopy(raw_schema)
    bad_range["classes"]["FieldModel"]["attributes"]["id"]["range"] = "NoSuchClass"
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_range, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="Unknown LinkML slot range 'NoSuchClass'"):
        compile_json_schema(tmp_path)

    bad_enum_const = deepcopy(raw_schema)
    bad_enum_const["classes"]["FrameworkModel"]["attributes"]["version"]["equals_string"] = "9.9-nonexistent"
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_enum_const, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="equals_string '9.9-nonexistent'.*is not in enum"):
        compile_json_schema(tmp_path)


def test_staffml_loader_rejects_duplicate_concepts_bad_placement_and_relation_drift(tmp_path: Path):
    import sys
    vault_cli_src = REPO_ROOT / "staffml/vault-cli/src"
    if str(vault_cli_src) not in sys.path:
        sys.path.insert(0, str(vault_cli_src))
    from vault_cli.ml_systems_taxonomy import load_taxonomy

    raw = load_taxonomy_yaml(REPO_ROOT / "taxonomy/ml-systems-taxonomy.yml")
    tax_dir = tmp_path / "taxonomy"
    schema_dir = tax_dir / "schema"
    vault_dir = tmp_path / "staffml/vault"
    schema_dir.mkdir(parents=True)
    vault_dir.mkdir(parents=True)
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(resolve_schema_path(REPO_ROOT).read_text())

    # 1. Duplicate concept ID rejected
    dup_conc = deepcopy(raw)
    dup_conc["concepts"].append(deepcopy(dup_conc["concepts"][0]))
    (tax_dir / "ml-systems-taxonomy.yml").write_text(yaml.dump(dup_conc, Dumper=yaml.SafeDumper, sort_keys=False))
    with pytest.raises(ValueError, match="duplicate concept"):
        load_taxonomy(vault_dir)

    # 2. Mismatched placement_examples home rejected
    bad_place = deepcopy(raw)
    bad_place["placement_examples"][0]["home"] = bad_place["areas"][0]["rows"][0]["id"]
    (tax_dir / "ml-systems-taxonomy.yml").write_text(yaml.dump(bad_place, Dumper=yaml.SafeDumper, sort_keys=False))
    with pytest.raises(ValueError, match="placement"):
        load_taxonomy(vault_dir)

    # 3. Drifted relation meaning rejected
    bad_rel = deepcopy(raw)
    bad_rel["relations"][0]["meaning"] = "drifted relation meaning"
    (tax_dir / "ml-systems-taxonomy.yml").write_text(yaml.dump(bad_rel, Dumper=yaml.SafeDumper, sort_keys=False))
    with pytest.raises(ValueError, match="relation .* disagrees"):
        load_taxonomy(vault_dir)


def test_load_taxonomy_yaml_rejects_duplicate_keys_and_aliases(tmp_path: Path):
    dup_file = tmp_path / "dup.yml"
    dup_file.write_text("a: 1\na: 2\n")
    with pytest.raises(ConceptGraphError, match="duplicate YAML mapping key: 'a'"):
        load_taxonomy_yaml(dup_file)

    alias_file = tmp_path / "alias.yml"
    alias_file.write_text("x: &anchor_ref [1, 2]\ny: *anchor_ref\n")
    with pytest.raises(ConceptGraphError, match="YAML anchors/aliases are not allowed"):
        load_taxonomy_yaml(alias_file)


def test_placement_examples_validated_against_concept_home(tmp_path: Path):
    raw = load_taxonomy_yaml(REPO_ROOT / "taxonomy/ml-systems-taxonomy.yml")
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    tax_dir = tmp_path / "taxonomy"
    schema_dir = tax_dir / "schema"
    schema_dir.mkdir(parents=True)
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(resolve_schema_path(REPO_ROOT).read_text())

    bad_example = deepcopy(raw)
    for area in bad_example["areas"]:
        for row in area["rows"]:
            row.pop("book_chapters", None)
    wrong_home = bad_example["areas"][0]["rows"][0]["id"]
    bad_example["placement_examples"][0]["home"] = wrong_home
    (tax_dir / "ml-systems-taxonomy.yml").write_text(
        yaml.dump(bad_example, Dumper=yaml.SafeDumper, sort_keys=False)
    )
    with pytest.raises(ConceptGraphError, match=f"placement example for .* claims home {wrong_home}"):
        load_framework(tmp_path, bundle["field"])


def test_public_release_downloads_validate_end_to_end():
    bundle = load_unified_taxonomy(REPO_ROOT)
    assert bundle is not None
    field = bundle["field"]
    framework = load_framework(REPO_ROOT, field)
    assert framework is not None
    rel_map = {
        k.name: v.decode()
        for k, v in release_artifacts(REPO_ROOT, framework, field).items()
        if "downloads" in k.parts
    }

    schema = json.loads(rel_map["mlsys-taxonomy.schema.json"])
    release = json.loads(rel_map["mlsys-taxonomy.json"])

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
    tree = ET.ElementTree(ET.fromstring(rel_map["mlsys-concept-graph.graphml"]))
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    nodes = tree.findall(".//g:node", ns)
    edges = tree.findall(".//g:edge", ns)
    assert len(nodes) == len(release["concepts"])
    assert len(edges) == len(release["relationships"])
    node_ids = {n.attrib["id"] for n in nodes}
    assert all(e.attrib["source"] in node_ids and e.attrib["target"] in node_ids for e in edges)

    # 3. Subjects CSV and Concepts CSV match the compiled JSON release.
    subj_rows = list(csv.DictReader(io.StringIO(rel_map["mlsys-taxonomy-subjects.csv"])))
    conc_rows = list(csv.DictReader(io.StringIO(rel_map["mlsys-taxonomy-concepts.csv"])))
    expected_subjects = sum(len(a["subjects"]) for g in release["groups"] for a in g["areas"])
    assert len(subj_rows) == expected_subjects
    assert len(conc_rows) == len(release["concepts"])

    # 4. SKOS Turtle (.ttl) contains all prefixes, 17 object properties, collections, and concepts.
    ttl_text = rel_map["mlsys-taxonomy.ttl"]
    assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#> ." in ttl_text
    assert ttl_text.count("a owl:ObjectProperty ;") == len(release["relations"])
    assert ttl_text.count("rdfs:subPropertyOf skos:related ;") == len(release["relations"])
    assert "owl:subPropertyOf" not in ttl_text
    assert ttl_text.count("a skos:Concept ;") == len(release["concepts"])


