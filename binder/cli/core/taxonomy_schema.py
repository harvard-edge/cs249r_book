"""LinkML metamodel compilation and ontological graph validation for the ML Systems Taxonomy.

This module bridges the declarative LinkML schema (`taxonomy/schema/mlsys-taxonomy.linkml.yml`)
with `binder concepts check` and `binder concepts build`. It enforces both:
1. Metamodel conformance (classes, slots, types, regex patterns, and controlled enums)
   compiled from the LinkML schema via `linkml`'s `SchemaView` and `JsonSchemaGenerator`
   (with a deterministic YAML-to-JSON-Schema fallback for minimal CI/secondary interpreters).
2. Ontological and semantic graph invariants (DAG acyclicity on taxonomic/mereological/dependency
   relations, verb-predicate polarity agreement, symptom bucket disjointness, and role coverage).
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

import jsonschema
import yaml

try:
    from linkml.generators.jsonschemagen import JsonSchemaGenerator
    from linkml_runtime.utils.schemaview import SchemaView
    HAS_LINKML = True
except ModuleNotFoundError:  # pragma: no cover
    JsonSchemaGenerator = None
    SchemaView = None
    HAS_LINKML = False

from .concept_graph import ConceptGraphError

SCHEMA_REL_PATH = Path("taxonomy/schema/mlsys-taxonomy.linkml.yml")
REPO_SCHEMA_PATH = Path(__file__).resolve().parents[3] / SCHEMA_REL_PATH

HIERARCHICAL_RELATIONS = ("specializes", "part-of", "requires")

DECREASING_VERBS = re.compile(
    r"\b(shrink|shrinks|shrinking|lower|lowers|lowering|reduce|reduces|reducing|decrease|decreases|decreasing|curb|curbs|curbing|cut|cuts|cutting|eliminate|eliminates|eliminating|minimize|minimizes|minimizing|compress|compresses|compressing)\b",
    re.IGNORECASE,
)
INCREASING_VERBS = re.compile(
    r"\b(increase|increases|increasing|raise|raises|raising|grow|grows|growing|inflate|inflates|inflating|amplify|amplifies|amplifying|worsen|worsens|worsening|exacerbate|exacerbates|exacerbating|multiply|multiplies|multiplying)\b",
    re.IGNORECASE,
)


def resolve_schema_path(root: Path) -> Path:
    """Return the LinkML schema path in `root` if present, otherwise the canonical repo schema."""
    candidate = root / SCHEMA_REL_PATH
    if candidate.exists():
        return candidate
    return REPO_SCHEMA_PATH


@lru_cache(maxsize=8)
def _cached_schema_view(schema_path_str: str, mtime_ns: int):
    del mtime_ns
    if HAS_LINKML and SchemaView is not None:
        return SchemaView(schema_path_str)
    return yaml.safe_load(Path(schema_path_str).read_text())


def get_schema_view(root: Path):
    """Load and cache the LinkML SchemaView for the taxonomy metamodel."""
    path = resolve_schema_path(root)
    return _cached_schema_view(str(path), path.stat().st_mtime_ns)


def _compile_linkml_fallback(schema_path_str: str, target_class: str | None) -> dict:
    """Deterministic LinkML-to-JSON-Schema compiler when running in a minimal interpreter."""
    raw = yaml.safe_load(Path(schema_path_str).read_text())
    enums = raw.get("enums", {})
    classes = raw.get("classes", {})
    defs: dict[str, dict] = {}
    for ename, edef in enums.items():
        defs[ename] = {
            "type": "string",
            " description": edef.get("description", ""),
            "enum": list((edef.get("permissible_values") or {}).keys()),
        }
        defs[ename].pop(" description", None)
    for cname, cdef in classes.items():
        props: dict[str, dict] = {}
        req: list[str] = []
        for sname, sdef in (cdef.get("attributes") or {}).items():
            sdef = sdef or {}
            rng = sdef.get("range", "string")
            inlined = bool(sdef.get("inlined") or sdef.get("inlined_as_list"))
            if rng == "integer":
                item_schema: dict = {"type": "integer"}
                if "equals_number" in sdef:
                    item_schema["const"] = sdef["equals_number"]
            elif rng == "boolean":
                item_schema = {"type": "boolean"}
            elif rng in enums:
                item_schema = {"$ref": f"#/$defs/{rng}"}
            elif rng == "AnyValue":
                item_schema = {"type": "object"}
            elif rng in classes and inlined:
                item_schema = {"$ref": f"#/$defs/{rng}"}
            else:
                item_schema = {"type": "string"}
            if "pattern" in sdef:
                item_schema["pattern"] = sdef["pattern"]
            if sdef.get("multivalued"):
                props[sname] = {"type": ["array", "null"], "items": item_schema}
            else:
                props[sname] = item_schema if sdef.get("required") else {"anyOf": [item_schema, {"type": "null"}]}
            if sdef.get("required"):
                req.append(sname)
        if cdef.get("class_uri") == "linkml:Any":
            defs[cname] = {}
        else:
            defs[cname] = {
                "type": "object",
                "properties": props,
                "required": req,
                "additionalProperties": False,
            }
    top = target_class or "TaxonomyReleaseDocument"
    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$id": raw.get("id", "https://mlsysbook.ai/taxonomy/schema"),
        "title": raw.get("title", "ML Systems Taxonomy Metamodel (LinkML)"),
        "$defs": defs,
        **(defs.get(top, {"type": "object"})),
    }


@lru_cache(maxsize=32)
def _cached_json_schema(schema_path_str: str, mtime_ns: int, target_class: str | None) -> tuple[dict, object]:
    del mtime_ns
    effective_top = target_class or "TaxonomyReleaseDocument"
    if HAS_LINKML and JsonSchemaGenerator is not None:
        gen = JsonSchemaGenerator(schema_path_str, top_class=effective_top, not_closed=False)
        schema_dict = json.loads(gen.serialize())
    else:
        schema_dict = _compile_linkml_fallback(schema_path_str, effective_top)
    validator_cls = jsonschema.validators.validator_for(schema_dict)
    validator_cls.check_schema(schema_dict)
    return schema_dict, validator_cls(schema_dict)


def compile_json_schema(root: Path, target_class: str | None = "TaxonomyReleaseDocument") -> dict:
    """Compile the LinkML metamodel (defaulting root to TaxonomyReleaseDocument) to a JSON Schema dict."""
    path = resolve_schema_path(root)
    schema_dict, _ = _cached_json_schema(str(path), path.stat().st_mtime_ns, target_class)
    return schema_dict


def validate_linkml(instance: object, target_class: str, root: Path, context: str) -> None:
    """Validate a loaded YAML document against its LinkML target class."""
    path = resolve_schema_path(root)
    if not path.exists():
        raise ConceptGraphError(f"{context}: missing LinkML schema at {path}")
    _, validator = _cached_json_schema(str(path), path.stat().st_mtime_ns, target_class)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        loc = ".".join(str(p) for p in first.absolute_path) or "<root>"
        raise ConceptGraphError(
            f"{context}: LinkML {target_class} validation failed at {loc}: {first.message}"
        )


def validate_field_semantics(model: dict) -> None:
    """Enforce ontological DAG acyclicity and verb-predicate polarity on field relationships."""
    edges = model.get("relationships", [])

    # 1. Ontological DAG acyclicity on hierarchical relations (specializes, part-of, requires).
    for rel_name in HIERARCHICAL_RELATIONS:
        adj: dict[str, list[str]] = {}
        for edge in edges:
            if edge.get("relation") == rel_name:
                adj.setdefault(edge["source"], []).append(edge["target"])
        state: dict[str, int] = {}  # 0/absent = unvisited, 1 = visiting, 2 = visited
        for start in list(adj):
            if state.get(start, 0) != 0:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            state[start] = 1
            while stack:
                node, idx = stack[-1]
                children = adj.get(node, [])
                if idx < len(children):
                    stack[-1] = (node, idx + 1)
                    nxt = children[idx]
                    nxt_state = state.get(nxt, 0)
                    if nxt_state == 1:
                        cycle_nodes = [n for n, _ in stack]
                        cycle_str = " -> ".join([*cycle_nodes[cycle_nodes.index(nxt):], nxt])
                        raise ConceptGraphError(
                            f"field model: ontological cycle in '{rel_name}' relation: {cycle_str}"
                        )
                    if nxt_state == 0:
                        state[nxt] = 1
                        stack.append((nxt, 0))
                else:
                    state[node] = 2
                    stack.pop()

    # 2. Verb-predicate polarity agreement on directional constraint/failure relations.
    for edge in edges:
        rel = edge.get("relation")
        predicate = edge.get("predicate", "")
        if rel == "increases" and DECREASING_VERBS.search(predicate) and not INCREASING_VERBS.search(predicate):
            raise ConceptGraphError(
                f"field model: relationship {edge['id']} predicate '{predicate}' contradicts relation 'increases'"
            )
        if rel in ("reduces", "mitigates") and INCREASING_VERBS.search(predicate) and not DECREASING_VERBS.search(predicate):
            raise ConceptGraphError(
                f"field model: relationship {edge['id']} predicate '{predicate}' contradicts relation '{rel}'"
            )


def validate_schema_sync(root: Path, field_model: dict | None = None) -> None:
    """Verify that LinkML enums stay synchronized with Python compiler constants and field relations."""
    path = resolve_schema_path(root)
    raw = yaml.safe_load(path.read_text())
    enums = {
        name: set((edef.get("permissible_values") or {}).keys())
        for name, edef in (raw.get("enums") or {}).items()
    }
    from .framework import SUPPORTED_VERSIONS
    from .taxonomy_coverage import STATUSES

    expected = {
        "ConceptScopeEnum": {
            "core", "background", "model-extension", "generative-extension",
            "agent-extension", "physical-extension", "specialist",
        },
        "RelationKindEnum": {
            "mechanism", "constraint", "tradeoff", "evaluation",
            "dependency", "failure", "composition",
        },
        "FrameworkVersionEnum": set(SUPPORTED_VERSIONS),
        "ChecklistStatusEnum": set(STATUSES),
    }
    for enum_name, exp_values in expected.items():
        actual = enums.get(enum_name, set())
        if actual != exp_values:
            raise ConceptGraphError(
                f"schema sync: LinkML {enum_name} {sorted(actual)} disagrees with compiler {sorted(exp_values)}"
            )
    if field_model and len(field_model.get("relations", [])) >= 17:
        rel_ids = {r["id"] for r in field_model["relations"]}
        if enums.get("RelationIdEnum", set()) != rel_ids:
            raise ConceptGraphError(
                f"schema sync: LinkML RelationIdEnum disagrees with ml-systems-field.yml relations"
            )


def validate_guides_semantics(guides: dict, rows_by_id: dict, version: str | None = None) -> None:
    """Enforce symptom bucket disjointness and (for 0.3-draft+) 100% subject role coverage."""
    for symptom in guides.get("symptoms", []):
        sid = symptom["id"]
        causes = set(symptom.get("causes", []))
        checks = set(symptom.get("checks", []))
        levers = set(symptom.get("levers", []))
        overlap_cl = sorted(causes & levers)
        if overlap_cl:
            raise ConceptGraphError(
                f"framework: symptom {sid} lists concept as both cause and lever: {', '.join(overlap_cl)}"
            )
        overlap_cc = sorted(causes & checks)
        if overlap_cc:
            raise ConceptGraphError(
                f"framework: symptom {sid} lists concept as both cause and check: {', '.join(overlap_cc)}"
            )
        overlap_kl = sorted(checks & levers)
        if overlap_kl:
            raise ConceptGraphError(
                f"framework: symptom {sid} lists concept as both check and lever: {', '.join(overlap_kl)}"
            )

    roles = guides.get("roles", [])
    if version not in (None, "0.1-draft", "0.2-draft") and len(rows_by_id) >= 20 and roles:
        covered = {
            subj
            for role in roles
            for subj in (*role.get("core", []), *role.get("working", []))
        }
        missing = sorted(set(rows_by_id) - covered)
        if missing:
            raise ConceptGraphError(
                f"framework: subjects missing from all role guides: {', '.join(missing)}"
            )
