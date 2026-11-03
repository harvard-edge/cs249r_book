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
    validate_term_provenance_sidecar,
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
    relations_text = rel_map["mlsys-taxonomy-relations.ttl"]
    assert relations_text.count("a owl:ObjectProperty") == len(release["relations"])
    assert relations_text.count("rdfs:subPropertyOf skos:broader ;") == 1  # only `specializes`; see Z39.19 8.3.3.2
    assert relations_text.count("rdfs:subPropertyOf skos:related ;") == len(release["relations"]) - 1
    assert "owl:subPropertyOf" not in ttl_text
    assert ttl_text.count("a skos:Concept ;") == len(release["concepts"]) + len(release["retired"])


def test_compiler_cache_immutability_and_fail_closed_metamodel_keys(tmp_path: Path):
    # Mutating returned schema dict must never poison the cached schema.
    first = compile_json_schema(REPO_ROOT)
    first["$defs"].clear()
    second = compile_json_schema(REPO_ROOT)
    assert "UnifiedTaxonomyModel" in second["$defs"]
    # Non-inlined class reference inherits target class identifier pattern.
    fam_prop = second["$defs"]["UnifiedConcept"]["properties"]["family"]
    assert fam_prop.get("pattern") == "^[a-z0-9]+(-[a-z0-9]+)*$"
    # Relationship.relation enforces RelationIdEnum at JSON Schema level.
    rel_prop = second["$defs"]["Relationship"]["properties"]["relation"]
    assert rel_prop == {"anyOf": [{"$ref": "#/$defs/RelationIdEnum"}, {"type": "null"}], "description": rel_prop["description"]}

    schema_dir = tmp_path / "taxonomy/schema"
    schema_dir.mkdir(parents=True)
    raw_schema = yaml.safe_load(resolve_schema_path(REPO_ROOT).read_text())

    bad_top = deepcopy(raw_schema)
    bad_top["slots"] = {"foo": {}}
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_top, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="Unsupported LinkML top-level keys: slots"):
        compile_json_schema(tmp_path)

    bad_cls = deepcopy(raw_schema)
    bad_cls["classes"]["FieldModel"]["is_a"] = "SomeParent"
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_cls, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="Unsupported LinkML class keys on FieldModel: is_a"):
        compile_json_schema(tmp_path)

    bad_enum = deepcopy(raw_schema)
    bad_enum["enums"]["RelationIdEnum"]["inherits"] = ["OtherEnum"]
    (schema_dir / "mlsys-taxonomy.linkml.yml").write_text(yaml.safe_dump(bad_enum, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="Unsupported LinkML enum keys on RelationIdEnum: inherits"):
        compile_json_schema(tmp_path)


def test_term_provenance_sidecar_binds_taxonomy_and_active_records(tmp_path: Path):
    import hashlib
    import re
    import shutil

    # 1. Verify post-1.0 patch version pattern (e.g. 1.0.1) in LinkML added_in regex.
    raw_schema = yaml.safe_load(resolve_schema_path(REPO_ROOT).read_text())
    for cls_name in ("Concept", "UnifiedConcept", "ReleaseConcept"):
        pat = raw_schema["classes"][cls_name]["attributes"]["added_in"]["pattern"]
        assert re.match(pat, "0.4-draft")
        assert re.match(pat, "1.0")
        assert re.match(pat, "1.0.1")
        assert re.match(pat, "1.2.3-rc1")
        assert not re.match(pat, "1.0-alpha")
    for cls_name in ("Concept", "ReleaseConcept"):
        desc = raw_schema["classes"][cls_name]["attributes"]["aliases"]["description"]
        assert "named system exemplars" not in desc

    # 2. Verify live repository provenance sidecar passes full schema_version=2 validation.
    validate_term_provenance_sidecar(REPO_ROOT)

    tax_path = REPO_ROOT / "taxonomy/ml-systems-taxonomy.yml"
    prov_path = REPO_ROOT / "taxonomy/term-provenance.yml"
    prov = load_taxonomy_yaml(prov_path)
    assert isinstance(prov, dict) and prov.get("schema_version") == 2

    # 3. Verify negative tamper checks fail closed via validate_term_provenance_sidecar.
    tdir = tmp_path / "taxonomy"
    tdir.mkdir(parents=True)
    shutil.copy2(tax_path, tdir / "ml-systems-taxonomy.yml")

    # 3a. Missing sidecar fails closed.
    with pytest.raises(ConceptGraphError, match="term-provenance.yml is missing"):
        validate_term_provenance_sidecar(tmp_path)

    # 3b. Hand-editing taxonomy_sha256 without updating sidecar_content_sha256 or applied_campaigns fails closed.
    shutil.copy2(prov_path, tdir / "term-provenance.yml")
    (tdir / "ml-systems-taxonomy.yml").write_text(tax_path.read_text() + "\n")
    new_sha = hashlib.sha256((tdir / "ml-systems-taxonomy.yml").read_bytes()).hexdigest()
    tampered = deepcopy(prov)
    tampered["taxonomy_sha256"] = new_sha
    (tdir / "term-provenance.yml").write_text(yaml.safe_dump(tampered, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="sidecar_content_sha256 mismatch"):
        validate_term_provenance_sidecar(tmp_path)

    # 3c. Even recomputing sidecar_content_sha256 without updating applied_campaigns post_apply_taxonomy_sha256 fails closed.
    tampered["non_guides_taxonomy_sha256"] = prov["non_guides_taxonomy_sha256"]
    body_no_self = {k: v for k, v in tampered.items() if k != "sidecar_content_sha256"}
    tampered["sidecar_content_sha256"] = hashlib.sha256(
        json.dumps(body_no_self, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (tdir / "term-provenance.yml").write_text(yaml.safe_dump(tampered, sort_keys=False))
    with pytest.raises(ConceptGraphError, match="post_apply_taxonomy_sha256 disagrees"):
        validate_term_provenance_sidecar(tmp_path)

    # 3d. Emptying concepts mapping while operations_applied lists governed concept updates fails closed.
    shutil.copy2(tax_path, tdir / "ml-systems-taxonomy.yml")
    empty_concepts = deepcopy(prov)
    empty_concepts["concepts"] = {}
    body2 = {k: v for k, v in empty_concepts.items() if k != "sidecar_content_sha256"}
    empty_concepts["sidecar_content_sha256"] = hashlib.sha256(
        json.dumps(body2, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (tdir / "term-provenance.yml").write_text(yaml.safe_dump(empty_concepts, sort_keys=False))
    with pytest.raises(
        ConceptGraphError,
        match="(missing provenance record|missing from term-provenance|unreconciled top-level provenance edit detected)",
    ):
        validate_term_provenance_sidecar(tmp_path)

    # 3e. Unpinned grandfathered_draft_alias_exceptions target (e.g. mlsys.stpa) fails closed.
    unpinned_gf = deepcopy(prov)
    unpinned_gf.setdefault("grandfathered_draft_alias_exceptions", {})["mlsys.stpa"] = {
        "aliases": ["UCA"],
        "status": "grandfathered-0.4-draft-queued-bucket-b",
        "reason": "Unpinned grandfathered target should be rejected.",
    }
    body3 = {k: v for k, v in unpinned_gf.items() if k != "sidecar_content_sha256"}
    unpinned_gf["sidecar_content_sha256"] = hashlib.sha256(
        json.dumps(body3, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (tdir / "term-provenance.yml").write_text(yaml.safe_dump(unpinned_gf, sort_keys=False))
    with pytest.raises(
        ConceptGraphError,
        match="(unpinned or inactive grandfathered_draft_alias_exceptions|unreconciled top-level provenance edit detected)",
    ):
        validate_term_provenance_sidecar(tmp_path)

    # 3f. Untracked edit to an untouched 0.4-draft concept (with recomputed self-digests) fails genesis replay.
    from core.taxonomy_schema import compute_campaign_entry_sha256, compute_non_guides_taxonomy_sha256

    raw_tax_tampered = load_taxonomy_yaml(tax_path)
    for cnode in raw_tax_tampered["concepts"]:
        if cnode["id"] == "mlsys.dataflow":
            cnode["definition"] = "Untracked alteration to a baseline 0.4-draft concept outside governed patches."
            break
    tampered_tax_txt = yaml.safe_dump(raw_tax_tampered, sort_keys=False, allow_unicode=True)
    (tdir / "ml-systems-taxonomy.yml").write_text(tampered_tax_txt)
    t_sha = hashlib.sha256(tampered_tax_txt.encode("utf-8")).hexdigest()
    t_ng_sha = compute_non_guides_taxonomy_sha256(raw_tax_tampered)
    forged_prov = deepcopy(prov)
    forged_prov["taxonomy_sha256"] = t_sha
    forged_prov["non_guides_taxonomy_sha256"] = t_ng_sha
    forged_prov["applied_campaigns"][-1]["post_apply_taxonomy_sha256"] = t_sha
    forged_prov["applied_campaigns"][-1]["post_non_guides_taxonomy_sha256"] = t_ng_sha
    forged_prov["applied_campaigns"][-1]["campaign_entry_sha256"] = compute_campaign_entry_sha256(
        forged_prov["applied_campaigns"][-1]
    )
    if forged_prov["applied_campaigns"][-1].get("mode") == "full-3family-reconcile":
        forged_prov["last_reconcile_receipt"] = deepcopy(forged_prov["applied_campaigns"][-1])
    body4 = {k: v for k, v in forged_prov.items() if k != "sidecar_content_sha256"}
    forged_prov["sidecar_content_sha256"] = hashlib.sha256(
        json.dumps(body4, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (tdir / "term-provenance.yml").write_text(yaml.safe_dump(forged_prov, sort_keys=False, allow_unicode=True))
    with pytest.raises(
        ConceptGraphError,
        match="(replayed non_guides_taxonomy_sha256 disagrees|replaying applied_campaigns from genesis)",
    ):
        validate_term_provenance_sidecar(tmp_path)


def test_float_overflow_1e999_and_non_finite_scalars_fail_closed(tmp_path: Path):
    """YAML and JSON loaders must reject non-finite scalars (.nan, .inf, 1e999 float overflow)."""
    from core.taxonomy_schema import json_loads_strict

    with pytest.raises(ConceptGraphError, match="non-finite"):
        json_loads_strict('{"alpha_subject": 1e999}', context="overflow_test")
    with pytest.raises(ConceptGraphError, match="non-finite"):
        json_loads_strict('{"alpha_subject": -1e999}', context="neg_overflow_test")

    overflow_yml = tmp_path / "overflow.yml"
    overflow_yml.write_text("id: mlsys-taxonomy\nscore: 1.0e+999\n")
    with pytest.raises(ConceptGraphError, match="non-finite"):
        load_taxonomy_yaml(overflow_yml)


def test_model_status_couples_to_draft_vs_normative_release():
    """LinkML ModelStatusEnum and replay_patch_on_taxonomy_doc couple status to -draft suffix."""
    from core.taxonomy_schema import replay_patch_on_taxonomy_doc

    pv = set(get_enum_values("ModelStatusEnum", REPO_ROOT))
    assert pv == {"editorial-draft", "normative-release"}

    doc = load_taxonomy_yaml(REPO_ROOT / "taxonomy" / "ml-systems-taxonomy.yml")
    assert doc["version"].endswith("-draft")
    assert doc["status"] == "editorial-draft"

    replay_patch_on_taxonomy_doc(doc, {"version": "1.0"}, mode="full-3family-reconcile")
    assert doc["version"] == "1.0"
    assert doc["status"] == "normative-release"
    assert doc["skos_hierarchy"]["status"] == "normative-release"

    replay_patch_on_taxonomy_doc(doc, {"version": "1.1-draft"}, mode="full-3family-reconcile")
    assert doc["version"] == "1.1-draft"
    assert doc["status"] == "editorial-draft"
    assert doc["skos_hierarchy"]["status"] == "editorial-draft"

