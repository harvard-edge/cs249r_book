"""LinkML metamodel compilation and ontological graph validation for the ML Systems Taxonomy.

This module bridges the declarative LinkML schema (`taxonomy/schema/mlsys-taxonomy.linkml.yml`)
with `binder concepts check` and `binder concepts build`. It enforces both:
1. Metamodel conformance (classes, slots, types, regex patterns, numeric bounds, and controlled enums)
   compiled from the LinkML schema into a deterministic Draft 2019-09 JSON Schema (while also
   verifying LinkML metamodel validity via `linkml_runtime.utils.schemaview.SchemaView` when installed).
2. Ontological and semantic graph invariants (DAG acyclicity on taxonomic/mereological/dependency
   relations, verb-predicate polarity agreement, symptom bucket disjointness, and role coverage).
"""
from __future__ import annotations

from functools import lru_cache
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

# Typed dynamic-key dictionary slots whose values must conform to specific LinkML classes.
_TYPED_DICT_SLOTS: dict[tuple[str, str], dict] = {
    ("AgreementRun", "taggers"): {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"$ref": "#/$defs/AgreementItemTag"},
        },
    },
    ("StaffMLLabelsDocument", "subjects"): {
        "type": "object",
        "propertyNames": {"pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]*$"},
        "additionalProperties": {"$ref": "#/$defs/StaffMLLabelSubject"},
    },
    ("StaffMLLabelsDocument", "concepts"): {
        "type": "object",
        "propertyNames": {"pattern": r"^mlsys\.[a-z0-9-]+$"},
        "additionalProperties": {"$ref": "#/$defs/StaffMLLabelConcept"},
    },
}


def resolve_schema_path(root: Path | None = None) -> Path:
    """Return the LinkML schema path in `root` if present, otherwise the canonical repo schema."""
    if root is not None:
        candidate = root / SCHEMA_REL_PATH
        if candidate.exists():
            return candidate
    return REPO_SCHEMA_PATH


@lru_cache(maxsize=8)
def _cached_raw_schema(schema_path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    return yaml.safe_load(Path(schema_path_str).read_text())


@lru_cache(maxsize=8)
def _cached_schema_view(schema_path_str: str, mtime_ns: int):
    del mtime_ns
    if HAS_LINKML and SchemaView is not None:
        return SchemaView(schema_path_str)
    return _cached_raw_schema(schema_path_str, Path(schema_path_str).stat().st_mtime_ns)


def get_schema_view(root: Path | None = None):
    """Load and cache the LinkML SchemaView for the taxonomy metamodel."""
    path = resolve_schema_path(root)
    return _cached_schema_view(str(path), path.stat().st_mtime_ns)


def get_enum_values(enum_name: str, root: Path | None = None) -> tuple[str, ...]:
    """Return the ordered tuple of permissible values for `enum_name` from the LinkML schema."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    enum_def = (raw.get("enums") or {}).get(enum_name) or {}
    return tuple((enum_def.get("permissible_values") or {}).keys())


@lru_cache(maxsize=8)
def _cached_base_schema(schema_path_str: str, mtime_ns: int) -> dict:
    """Compile the LinkML metamodel into a canonical Draft 2019-09 JSON Schema `$defs` bundle once per file mtime."""
    raw = _cached_raw_schema(schema_path_str, mtime_ns)
    if HAS_LINKML and SchemaView is not None:
        # Validate the LinkML metamodel itself when linkml-runtime is installed.
        _cached_schema_view(schema_path_str, mtime_ns)

    enums = raw.get("enums") or {}
    classes = raw.get("classes") or {}
    defs: dict[str, dict] = {}
    for ename, edef in enums.items():
        edef = edef or {}
        entry: dict = {
            "type": "string",
            "enum": list((edef.get("permissible_values") or {}).keys()),
        }
        if edef.get("description"):
            entry["description"] = edef["description"]
        defs[ename] = entry

    for cname, cdef in classes.items():
        cdef = cdef or {}
        if cdef.get("class_uri") == "linkml:Any":
            defs[cname] = {
                "description": cdef.get("description", "Arbitrary mapping value."),
                "type": "object",
            }
            continue
        props: dict[str, dict] = {}
        req: list[str] = []
        for sname, sdef in (cdef.get("attributes") or {}).items():
            sdef = sdef or {}
            rng = sdef.get("range", "string")
            inlined = bool(sdef.get("inlined") or sdef.get("inlined_as_list"))
            if (cname, sname) in _TYPED_DICT_SLOTS:
                item_schema = dict(_TYPED_DICT_SLOTS[(cname, sname)])
            elif rng == "integer":
                item_schema = {"type": "integer"}
                if "equals_number" in sdef:
                    item_schema["const"] = sdef["equals_number"]
                if "minimum_value" in sdef:
                    item_schema["minimum"] = sdef["minimum_value"]
                if "maximum_value" in sdef:
                    item_schema["maximum"] = sdef["maximum_value"]
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
                if "equals_string" in sdef:
                    item_schema["const"] = sdef["equals_string"]
            if "pattern" in sdef:
                item_schema["pattern"] = sdef["pattern"]
            if sdef.get("multivalued"):
                prop_schema: dict = (
                    {"type": "array", "items": item_schema}
                    if sdef.get("required")
                    else {"type": ["array", "null"], "items": item_schema}
                )
            else:
                prop_schema = (
                    dict(item_schema)
                    if sdef.get("required")
                    else {"anyOf": [item_schema, {"type": "null"}]}
                )
            if sdef.get("description"):
                prop_schema["description"] = sdef["description"]
            props[sname] = prop_schema
            if sdef.get("required"):
                req.append(sname)
        cls_schema: dict = {
            "type": "object",
            "properties": props,
            "required": req,
            "additionalProperties": False,
        }
        if cdef.get("description"):
            cls_schema["description"] = cdef["description"]
        defs[cname] = cls_schema

    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "$id": raw.get("id", "https://mlsysbook.ai/taxonomy/schema"),
        "title": raw.get("title", "ML Systems Taxonomy Metamodel (LinkML)"),
        "version": str(raw.get("version", "")),
        "$defs": defs,
    }


@lru_cache(maxsize=32)
def _cached_json_schema(schema_path_str: str, mtime_ns: int, target_class: str | None) -> tuple[dict, object]:
    base = _cached_base_schema(schema_path_str, mtime_ns)
    effective_top = target_class or "TaxonomyReleaseDocument"
    top_def = base["$defs"].get(effective_top, {"type": "object"})
    schema_dict = {
        "$schema": base["$schema"],
        "$id": base["$id"],
        "title": base["title"],
        "$defs": base["$defs"],
        **top_def,
    }
    validator_cls = jsonschema.validators.validator_for(schema_dict)
    validator_cls.check_schema(schema_dict)
    return schema_dict, validator_cls(schema_dict)


def compile_json_schema(root: Path, target_class: str | None = "TaxonomyReleaseDocument") -> dict:
    """Compile the LinkML metamodel (defaulting root to TaxonomyReleaseDocument) to a deterministic JSON Schema dict."""
    path = resolve_schema_path(root)
    schema_dict, _ = _cached_json_schema(str(path), path.stat().st_mtime_ns, target_class)
    return schema_dict


def _format_error_path(instance: object, path_steps: list) -> str:
    """Format a jsonschema error path, annotating list indices with their record `id` when available."""
    if not path_steps:
        return "<root>"
    parts: list[str] = []
    curr = instance
    for step in path_steps:
        if isinstance(step, int) and isinstance(curr, list) and 0 <= step < len(curr):
            elem = curr[step]
            rec_id = (
                elem.get("id") or elem.get("inventory_id") or elem.get("topic") or elem.get("n")
                if isinstance(elem, dict)
                else None
            )
            label = f"[{step}:{rec_id}]" if rec_id is not None else f"[{step}]"
            if parts:
                parts[-1] = f"{parts[-1]}{label}"
            else:
                parts.append(label)
            curr = elem
        else:
            parts.append(str(step))
            curr = curr.get(step) if isinstance(curr, dict) else None
    return ".".join(parts)


def validate_linkml(instance: object, target_class: str, root: Path, context: str) -> None:
    """Validate a loaded YAML document against its LinkML target class."""
    path = resolve_schema_path(root)
    if not path.exists():
        raise ConceptGraphError(f"{context}: missing LinkML schema at {path}")
    _, validator = _cached_json_schema(str(path), path.stat().st_mtime_ns, target_class)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        formatted = [
            f"{_format_error_path(instance, list(err.absolute_path))}: {err.message}"
            for err in errors[:3]
        ]
        more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        raise ConceptGraphError(
            f"{context}: LinkML {target_class} validation failed at {'; '.join(formatted)}{more}"
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
    """Verify that LinkML enums and version stay synchronized with Python compiler constants and field relations."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    enums = {
        name: set((edef.get("permissible_values") or {}).keys())
        for name, edef in (raw.get("enums") or {}).items()
    }
    from .framework import SUPPORTED_VERSIONS
    from .taxonomy_coverage import STATUSES

    expected = {
        "ConceptScopeEnum": set(get_enum_values("ConceptScopeEnum", root)),
        "RelationKindEnum": set(get_enum_values("RelationKindEnum", root)),
        "FrameworkVersionEnum": set(SUPPORTED_VERSIONS),
        "ChecklistStatusEnum": set(STATUSES),
    }
    for enum_name, exp_values in expected.items():
        actual = enums.get(enum_name, set())
        if actual != exp_values:
            raise ConceptGraphError(
                f"schema sync: LinkML {enum_name} {sorted(actual)} disagrees with compiler {sorted(exp_values)}"
            )
    schema_ver = str(raw.get("version", ""))
    if SUPPORTED_VERSIONS and schema_ver != SUPPORTED_VERSIONS[-1]:
        raise ConceptGraphError(
            f"schema sync: LinkML top-level version '{schema_ver}' disagrees with latest FrameworkVersionEnum '{SUPPORTED_VERSIONS[-1]}'"
        )
    if field_model and len(field_model.get("relations", [])) >= 17:
        rel_ids = {r["id"] for r in field_model["relations"]}
        if enums.get("RelationIdEnum", set()) != rel_ids:
            raise ConceptGraphError(
                "schema sync: LinkML RelationIdEnum disagrees with ml-systems-taxonomy.yml relations"
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
