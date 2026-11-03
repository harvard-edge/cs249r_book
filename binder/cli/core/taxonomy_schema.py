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

import collections
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from urllib.parse import urlparse

import jsonschema
import yaml

try:
    from linkml_runtime.utils.schemaview import SchemaView
    HAS_LINKML = True
except ModuleNotFoundError:  # pragma: no cover
    SchemaView = None
    HAS_LINKML = False

from .concept_graph import ConceptGraphError

SCHEMA_REL_PATH = Path("taxonomy/schema/mlsys-taxonomy.linkml.yml")
REPO_SCHEMA_PATH = Path(__file__).resolve().parents[3] / SCHEMA_REL_PATH

CONCEPT_ID_RE = re.compile(r"^mlsys\.[a-z0-9]+(-[a-z0-9]+)*$")
SCOPE_REF_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_UNSAFE_URL_CHARS = re.compile(r'[\s<>"{}|\\^`]')

DECREASING_VERBS = re.compile(
    r"\b(shrink|shrinks|shrinking|lower|lowers|lowering|reduce|reduces|reducing|decrease|decreases|decreasing|curb|curbs|curbing|cut|cuts|cutting|eliminate|eliminates|eliminating|minimize|minimizes|minimizing|compress|compresses|compressing|invalidate|invalidates|invalidating)\b",
    re.IGNORECASE,
)
INCREASING_VERBS = re.compile(
    r"\b(increase|increases|increasing|raise|raises|raising|grow|grows|growing|inflate|inflates|inflating|amplify|amplifies|amplifying|worsen|worsens|worsening|exacerbate|exacerbates|exacerbating|multiply|multiplies|multiplying)\b",
    re.IGNORECASE,
)


def require_text(
    record: dict,
    keys: tuple[str, ...],
    context: str,
    allowed: set[str] | None = None,
) -> dict[str, str]:
    """Validate that `record` is a mapping with non-empty strings for `keys` and no unknown keys."""
    if not isinstance(record, dict):
        raise ConceptGraphError(f"{context} must be a mapping")
    extra = set(record) - (allowed or set(keys))
    if extra:
        raise ConceptGraphError(f"{context} has unknown keys: {', '.join(sorted(extra))}")
    out: dict[str, str] = {}
    for key in keys:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConceptGraphError(f"{context}.{key} must be a non-empty string")
        out[key] = value.strip()
    return out


def validate_https_url(url: str, error_message: str) -> str:
    """Validate that `url` is a clean HTTPS URL without credentials or unsafe characters."""
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or _UNSAFE_URL_CHARS.search(url)
    ):
        raise ConceptGraphError(error_message)
    return url

# LinkML keys supported by the deterministic Draft 2019-09 JSON Schema compiler.
# Any unknown key at top-level, class-level, enum-level, or slot-level raises ConceptGraphError
# so unsupported LinkML directives (is_a, mixins, rules, slot_usage, etc.) fail closed.
_ALLOWED_TOP_KEYS = frozenset({
    "classes",
    "default_prefix",
    "default_range",
    "description",
    "enums",
    "id",
    "imports",
    "license",
    "name",
    "prefixes",
    "title",
    "version",
})
_ALLOWED_CLASS_KEYS = frozenset({
    "attributes",
    "class_uri",
    "description",
    "tree_root",
})
_ALLOWED_ENUM_KEYS = frozenset({
    "description",
    "permissible_values",
})
_ALLOWED_PV_KEYS = frozenset({
    "annotations",
    "description",
})
_ALLOWED_SLOT_KEYS = frozenset({
    "annotations",
    "description",
    "equals_number",
    "equals_string",
    "identifier",
    "inlined",
    "inlined_as_list",
    "maximum_cardinality",
    "maximum_value",
    "minimum_cardinality",
    "minimum_value",
    "multivalued",
    "pattern",
    "range",
    "required",
    "slot_uri",
})
_ALLOWED_ANNOTATION_KEYS = frozenset({
    "dict_key_pattern",
    "dict_value_array_ref",
    "dict_value_ref",
})


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
    if HAS_LINKML and SchemaView is not None:
        return SchemaView(schema_path_str)
    return _cached_raw_schema(schema_path_str, mtime_ns)


def get_schema_view(root: Path | None = None):
    """Load and cache the LinkML SchemaView for the taxonomy metamodel."""
    path = resolve_schema_path(root)
    view = _cached_schema_view(str(path), path.stat().st_mtime_ns)
    return deepcopy(view) if isinstance(view, dict) else view


def get_enum_values(enum_name: str, root: Path | None = None) -> tuple[str, ...]:
    """Return the ordered tuple of permissible values for `enum_name` from the LinkML schema."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    enum_def = (raw.get("enums") or {}).get(enum_name) or {}
    return tuple((enum_def.get("permissible_values") or {}).keys())


def get_class_slots(class_name: str, root: Path | None = None) -> tuple[str, ...]:
    """Return the ordered tuple of attribute slot names declared on `class_name` in the LinkML schema."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    cls_def = (raw.get("classes") or {}).get(class_name)
    if not cls_def:
        raise ConceptGraphError(f"Unknown LinkML class: {class_name}")
    return tuple((cls_def.get("attributes") or {}).keys())


def get_required_class_slots(class_name: str, root: Path | None = None) -> tuple[str, ...]:
    """Return the ordered tuple of required attribute slot names declared on `class_name`."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    cls_def = (raw.get("classes") or {}).get(class_name)
    if not cls_def:
        raise ConceptGraphError(f"Unknown LinkML class: {class_name}")
    attrs = cls_def.get("attributes") or {}
    return tuple(name for name, sdef in attrs.items() if isinstance(sdef, dict) and sdef.get("required"))


def get_hierarchical_relations(root: Path | None = None) -> tuple[str, ...]:
    """Return the tuple of RelationIdEnum values annotated with `hierarchical_dag: true` in the LinkML schema."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    pvs = ((raw.get("enums") or {}).get("RelationIdEnum") or {}).get("permissible_values") or {}
    return tuple(
        rel_id
        for rel_id, pv in pvs.items()
        if isinstance(pv, dict) and isinstance(pv.get("annotations"), dict) and pv["annotations"].get("hierarchical_dag") is True
    )


HIERARCHICAL_RELATIONS = get_hierarchical_relations()


@lru_cache(maxsize=8)
def _cached_base_schema(schema_path_str: str, mtime_ns: int) -> dict:
    """Compile the LinkML metamodel into a canonical Draft 2019-09 JSON Schema `$defs` bundle once per file mtime."""
    raw = _cached_raw_schema(schema_path_str, mtime_ns)
    if not isinstance(raw, dict):
        raise ConceptGraphError("LinkML schema root must be a mapping")
    unknown_top = set(raw) - _ALLOWED_TOP_KEYS
    if unknown_top:
        raise ConceptGraphError(f"Unsupported LinkML top-level keys: {', '.join(sorted(unknown_top))}")
    if raw.get("default_range", "string") != "string":
        raise ConceptGraphError(f"Unsupported LinkML default_range: {raw.get('default_range')}")
    enums = raw.get("enums") or {}
    classes = raw.get("classes") or {}
    defs: dict[str, dict] = {}
    for ename, edef in enums.items():
        edef = edef or {}
        unknown_enum_keys = set(edef) - _ALLOWED_ENUM_KEYS
        if unknown_enum_keys:
            raise ConceptGraphError(
                f"Unsupported LinkML enum keys on {ename}: {', '.join(sorted(unknown_enum_keys))}"
            )
        pvs = edef.get("permissible_values") or {}
        for pv_name, pv_def in pvs.items():
            if isinstance(pv_def, dict):
                unknown_pv_keys = set(pv_def) - _ALLOWED_PV_KEYS
                if unknown_pv_keys:
                    raise ConceptGraphError(
                        f"Unsupported LinkML permissible_value keys on {ename}.{pv_name}: {', '.join(sorted(unknown_pv_keys))}"
                    )
        entry: dict = {
            "type": "string",
            "enum": list(pvs.keys()),
        }
        if edef.get("description"):
            entry["description"] = edef["description"]
        defs[ename] = entry

    for cname, cdef in classes.items():
        cdef = cdef or {}
        unknown_class_keys = set(cdef) - _ALLOWED_CLASS_KEYS
        if unknown_class_keys:
            raise ConceptGraphError(
                f"Unsupported LinkML class keys on {cname}: {', '.join(sorted(unknown_class_keys))}"
            )
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
            unknown_keys = set(sdef) - _ALLOWED_SLOT_KEYS
            if unknown_keys:
                raise ConceptGraphError(
                    f"Unsupported LinkML slot keys on {cname}.{sname}: {', '.join(sorted(unknown_keys))}"
                )
            annotations = sdef.get("annotations") or {}
            if not isinstance(annotations, dict) or (set(annotations) - _ALLOWED_ANNOTATION_KEYS):
                bad_ann = sorted(set(annotations) - _ALLOWED_ANNOTATION_KEYS) if isinstance(annotations, dict) else [str(annotations)]
                raise ConceptGraphError(
                    f"Unsupported LinkML slot annotations on {cname}.{sname}: {', '.join(bad_ann)}"
                )
            rng = sdef.get("range", "string")
            inlined = bool(sdef.get("inlined") or sdef.get("inlined_as_list"))
            if rng not in enums and rng not in classes and rng not in ("string", "integer", "boolean"):
                raise ConceptGraphError(f"Unknown LinkML slot range '{rng}' on {cname}.{sname}")
            is_string_like = (not annotations) and (
                rng == "string" or rng in enums or (rng in classes and not inlined and rng != "AnyValue")
            )
            is_integer = (not annotations) and rng == "integer"
            if inlined and (annotations or rng not in classes or rng == "AnyValue"):
                raise ConceptGraphError(f"inlined/inlined_as_list on non-class slot {cname}.{sname} is not allowed")
            for num_key in ("equals_number", "minimum_value", "maximum_value"):
                if num_key in sdef:
                    if not is_integer or not isinstance(sdef[num_key], int) or isinstance(sdef[num_key], bool):
                        raise ConceptGraphError(f"{num_key} on non-integer slot {cname}.{sname} is not allowed")
            for str_key in ("equals_string", "pattern"):
                if str_key in sdef:
                    if not is_string_like or not isinstance(sdef[str_key], str):
                        raise ConceptGraphError(f"{str_key} on non-string slot {cname}.{sname} is not allowed")
            if annotations:
                item_schema: dict = {"type": "object"}
                if "dict_key_pattern" in annotations:
                    item_schema["propertyNames"] = {"pattern": annotations["dict_key_pattern"]}
                if "dict_value_ref" in annotations:
                    ref_cls = annotations["dict_value_ref"]
                    if ref_cls not in classes:
                        raise ConceptGraphError(f"Unknown dict_value_ref class {ref_cls} on {cname}.{sname}")
                    item_schema["additionalProperties"] = {"$ref": f"#/$defs/{ref_cls}"}
                elif "dict_value_array_ref" in annotations:
                    ref_cls = annotations["dict_value_array_ref"]
                    if ref_cls not in classes:
                        raise ConceptGraphError(f"Unknown dict_value_array_ref class {ref_cls} on {cname}.{sname}")
                    item_schema["additionalProperties"] = {
                        "type": "array",
                        "items": {"$ref": f"#/$defs/{ref_cls}"},
                    }
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
                if "equals_string" in sdef or "pattern" in sdef:
                    pvs = (enums[rng] or {}).get("permissible_values") or {}
                    if "equals_string" in sdef and sdef["equals_string"] not in pvs:
                        raise ConceptGraphError(
                            f"equals_string '{sdef['equals_string']}' on {cname}.{sname} is not in enum {rng}"
                        )
                    sub: dict = {}
                    if "equals_string" in sdef:
                        sub["const"] = sdef["equals_string"]
                    if "pattern" in sdef:
                        sub["pattern"] = sdef["pattern"]
                    item_schema = {"allOf": [{"$ref": f"#/$defs/{rng}"}, sub]}
                else:
                    item_schema = {"$ref": f"#/$defs/{rng}"}
            elif rng == "AnyValue":
                item_schema = {"type": "object"}
            elif rng in classes and inlined:
                item_schema = {"$ref": f"#/$defs/{rng}"}
            else:
                item_schema = {"type": "string"}
                if rng in classes:
                    target_attrs = (classes[rng] or {}).get("attributes") or {}
                    for tsdef in target_attrs.values():
                        if isinstance(tsdef, dict) and tsdef.get("identifier") and isinstance(tsdef.get("pattern"), str):
                            item_schema["pattern"] = tsdef["pattern"]
                            break
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
                if "minimum_cardinality" in sdef:
                    prop_schema["minItems"] = sdef["minimum_cardinality"]
                if "maximum_cardinality" in sdef:
                    prop_schema["maxItems"] = sdef["maximum_cardinality"]
            else:
                if "minimum_cardinality" in sdef or "maximum_cardinality" in sdef:
                    raise ConceptGraphError(
                        f"Cardinality bounds on non-multivalued slot {cname}.{sname} are not allowed"
                    )
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

    if HAS_LINKML and SchemaView is not None:
        # Validate the LinkML metamodel itself when linkml-runtime is installed.
        _cached_schema_view(schema_path_str, mtime_ns)

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
    if effective_top not in base["$defs"]:
        raise ConceptGraphError(f"Unknown LinkML target class: {effective_top}")
    top_def = base["$defs"][effective_top]
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
    return deepcopy(schema_dict)


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
    """Enforce referential integrity, ontological DAG acyclicity, and verb-predicate polarity on field relationships."""
    edges = model.get("relationships", [])
    concepts_list = model.get("concepts")
    families_list = model.get("families")
    sources_list = model.get("sources")
    source_ids = (
        {s["id"] for s in sources_list if isinstance(s, dict) and isinstance(s.get("id"), str)}
        if isinstance(sources_list, list) and sources_list
        else None
    )
    if "concepts" in model and not isinstance(concepts_list, list):
        raise ConceptGraphError("field model: concepts must be a list")
    if isinstance(concepts_list, list):
        concept_ids = {c["id"] for c in concepts_list if isinstance(c, dict) and isinstance(c.get("id"), str)}
        if isinstance(families_list, list) and families_list:
            family_ids = {f["id"] for f in families_list if isinstance(f, dict) and isinstance(f.get("id"), str)}
            if source_ids is not None:
                for f in families_list:
                    if isinstance(f, dict):
                        refs = f.get("references") or []
                        if not isinstance(refs, list) or not refs or any(r not in source_ids for r in refs):
                            raise ConceptGraphError(f"field model: {f.get('id')} requires known reading references")
            for c in concepts_list:
                if isinstance(c, dict) and "family" in c and c.get("family") not in family_ids:
                    raise ConceptGraphError(f"field model: invalid concept identity or family {c.get('id')}")
        if source_ids is not None:
            for c in concepts_list:
                if isinstance(c, dict) and "references" in c:
                    refs = c.get("references") or []
                    if not isinstance(refs, list) or not refs or any(r not in source_ids for r in refs):
                        raise ConceptGraphError(f"field model: {c.get('id')} requires known reading references")
        for edge in edges:
            if isinstance(edge, dict):
                src = edge.get("source")
                tgt = edge.get("target")
                if src not in concept_ids or tgt not in concept_ids or src == tgt:
                    raise ConceptGraphError(f"field model: missing or identical endpoints {edge.get('id')}")
                if source_ids is not None and "references" in edge:
                    refs = edge.get("references") or []
                    if not isinstance(refs, list) or not refs or any(r not in source_ids for r in refs):
                        raise ConceptGraphError(f"field model: {edge.get('id')} requires known reading references")
        for record in model.get("retired", []) or []:
            if isinstance(record, dict):
                if record.get("id") in concept_ids or record.get("replaced_by") not in concept_ids:
                    raise ConceptGraphError(
                        f"field model: retired identity {record.get('id')} must be unused and name a current replacement"
                    )

    # 0. SKOS Reference S27 (2026-11-02): `specializes` exports as skos:broader, so no other relation may join a
    # concept to any of its transitive ancestors; skos:related and skos:broaderTransitive are disjoint.
    parents: dict[str, set[str]] = {}
    well_formed = [e for e in edges if isinstance(e, dict) and "source" in e and "target" in e and "id" in e]
    for edge in well_formed:
        if edge.get("relation") == "specializes":
            parents.setdefault(edge["source"], set()).add(edge["target"])

    def _ancestors(node: str) -> set[str]:
        seen: set[str] = set()
        todo = list(parents.get(node, ()))
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo.extend(parents.get(current, ()))
        return seen

    for edge in well_formed:
        if edge.get("relation") != "specializes" and (
            edge["target"] in _ancestors(edge["source"]) or edge["source"] in _ancestors(edge["target"])
        ):
            raise ConceptGraphError(
                f"field model: {edge['id']} joins {edge['source']} to its `specializes` ancestor or descendant "
                f"{edge['target']}; SKOS S27 keeps associative and hierarchical links disjoint"
            )

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
        if rel in ("reduces", "mitigates", "limits") and INCREASING_VERBS.search(predicate) and not DECREASING_VERBS.search(predicate):
            raise ConceptGraphError(
                f"field model: relationship {edge['id']} predicate '{predicate}' contradicts relation '{rel}'"
            )


def validate_schema_sync(root: Path, field_model: dict | None = None) -> None:
    """Verify that LinkML enums, relation metadata, version, and workflow briefs stay synchronized."""
    path = resolve_schema_path(root)
    raw = _cached_raw_schema(str(path), path.stat().st_mtime_ns)
    raw_enums = raw.get("enums") or {}
    enums = {
        name: set((edef.get("permissible_values") or {}).keys())
        for name, edef in raw_enums.items()
    }
    from .taxonomy_coverage import STATUSES

    expected = {
        "ChecklistStatusEnum": set(STATUSES),
    }
    for enum_name, exp_values in expected.items():
        actual = enums.get(enum_name, set())
        if actual != exp_values:
            raise ConceptGraphError(
                f"schema sync: LinkML {enum_name} {sorted(actual)} disagrees with compiler {sorted(exp_values)}"
            )
    fw_versions = [
        str(k) for k in ((raw_enums.get("FrameworkVersionEnum") or {}).get("permissible_values") or {}).keys()
    ]
    if not fw_versions or any(not _SEMVER_RE.fullmatch(v) for v in fw_versions):
        raise ConceptGraphError(f"schema sync: invalid or empty LinkML FrameworkVersionEnum {fw_versions}")
    for i in range(len(fw_versions) - 1):
        if parse_taxonomy_semver(fw_versions[i]) >= parse_taxonomy_semver(fw_versions[i + 1]):
            raise ConceptGraphError(
                f"schema sync: LinkML FrameworkVersionEnum must be monotonically increasing in SemVer order: {fw_versions}"
            )
    schema_ver = str(raw.get("version", ""))
    latest_fw_ver = max(fw_versions, key=parse_taxonomy_semver)
    if schema_ver != latest_fw_ver:
        raise ConceptGraphError(
            f"schema sync: LinkML top-level version '{schema_ver}' disagrees with latest FrameworkVersionEnum '{latest_fw_ver}'"
        )

    rel_pvs = (raw_enums.get("RelationIdEnum") or {}).get("permissible_values") or {}
    if field_model and "relations" in field_model:
        relations = field_model.get("relations") or []
        rel_ids = {r["id"] for r in relations if isinstance(r, dict) and "id" in r}
        if set(rel_pvs.keys()) != rel_ids or len(relations) != len(rel_pvs):
            raise ConceptGraphError(
                "schema sync: LinkML RelationIdEnum disagrees with ml-systems-taxonomy.yml relations"
            )
        for rel in relations:
            rid = rel["id"]
            pv = rel_pvs.get(rid) or {}
            ann = pv.get("annotations") or {}
            if rel.get("meaning") != pv.get("description"):
                raise ConceptGraphError(
                    f"schema sync: RelationIdEnum.{rid} description disagrees with ml-systems-taxonomy.yml meaning"
                )
            for field_key in ("kind", "forward", "inverse"):
                if rel.get(field_key) != ann.get(field_key):
                    raise ConceptGraphError(
                        f"schema sync: RelationIdEnum.{rid} annotation '{field_key}' disagrees with ml-systems-taxonomy.yml"
                    )

    brief_path = root / ".claude/workflows/taxonomy-refresh/briefs/common.md"
    if brief_path.exists():
        brief_text = brief_path.read_text()
        match = re.search(
            r"Relation vocabulary \(use only these (\d+)\):\s*([a-z0-9,\s-]+)\.",
            brief_text,
        )
        if not match:
            raise ConceptGraphError("schema sync: common.md is missing the Relation vocabulary declaration")
        declared_count = int(match.group(1))
        declared_verbs = [v.strip() for v in match.group(2).replace("\n", " ").split(",") if v.strip()]
        if declared_count != len(rel_pvs) or set(declared_verbs) != set(rel_pvs.keys()):
            raise ConceptGraphError(
                "schema sync: .claude/workflows/taxonomy-refresh/briefs/common.md relation vocabulary disagrees with RelationIdEnum"
            )

    readme_path = root / "taxonomy/README.md"
    if (
        readme_path.exists()
        and field_model
        and field_model.get("id") == "ml-systems-field"
        and "concepts" in field_model
        and "relationships" in field_model
    ):
        readme_text = readme_path.read_text()
        n_concepts = len(field_model["concepts"])
        n_rels = f"{len(field_model['relationships']):,}"
        if f"the {n_concepts} concepts" not in readme_text or f"has {n_rels} relationships" not in readme_text:
            raise ConceptGraphError(
                f"schema sync: taxonomy/README.md counts disagree with ml-systems-taxonomy.yml ({n_concepts} concepts, {n_rels} relationships)"
            )

    prov_path = root / "taxonomy/term-provenance.yml"
    if prov_path.exists() or readme_path.exists():
        validate_term_provenance_sidecar(root)


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(?:-(draft|rc([0-9]+)))?$")
_PRE_SIDECAR_VERSIONS = {"0.1-draft", "0.2-draft", "0.3-draft", "0.4-draft"}
_ALLOWED_PROV_ALIAS_TYPES = {"synonym", "quasi-synonym", "acronym", "upward-posting"}
_ALLOWED_OP_PREFIXES = (
    "add:",
    "update:",
    "retire:",
    "move:",
    "edge:",
    "rel_update:",
    "rel_remove:",
    "gap_add:",
    "gap_update:",
    "gap_remove:",
    "row:",
    "pe:",
    "guides:",
)
_ALLOWED_GUIDES_OP_PREFIXES = (
    "role_add:",
    "role_update:",
    "symptom_add:",
    "symptom_update:",
    "path_add:",
    "path_update:",
)

GENESIS_SNAPSHOT_REL_PATH = Path("taxonomy/method-study/genesis-0.4-baseline.json.gz")
BURNED_CLUSTERS_REL_PATH = Path("taxonomy/method-study/burned-clusters.json")
REPO_ROOT_PATH = Path(__file__).resolve().parents[3]

GENESIS_PRE_SIDECAR_TAXONOMY_SHA256 = "a3eefe42c766e031c43430aba061a56605c94117decf459b83962213f01de39e"
GENESIS_PRE_SIDECAR_NON_GUIDES_SHA256 = "1fd44b93987fea8ec835414c04cb8417cf127ae5d0ff4fbfb7fb9ade6ba934cc"
GENESIS_SNAPSHOT_GZ_SHA256 = "2b49f250a20c758d4cd46b32ab37b149baf395aaae50b1d5ab7d982a21f6ff09"
GENESIS_CANONICAL_JSON_SHA256 = "22b889e18a707601365572779bedff904f8a609d007a79d1ec62aadb7281a056"
GENESIS_HISTORICAL_PHASE3_UNITS_SHA256 = "da9c70a8dd6a4cb9272806ff4748b72a7da46e66b2a38b5c91144091451db279"

PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "mlsys.safety-case": ("assurance case",),
    "mlsys.tree-ensemble": ("XGBoost",),
    "mlsys.constrained-decoding": ("grammar-guided decoding",),
    "mlsys.state-sharding": ("ZeRO", "FSDP"),
    "mlsys.paged-kv": ("PagedAttention",),
    "mlsys.graph-replay": ("CUDA Graphs",),
    "mlsys.device-streams": ("CUDA streams",),
}

BOOTSTRAP_GRANDFATHERED_ALIAS_EXCEPTIONS: dict[str, dict[str, object]] = {
    "mlsys.safety-case": {
        "aliases": ["assurance case"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft broader cross-domain term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.tree-ensemble": {
        "aliases": ["XGBoost"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named implementation entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.constrained-decoding": {
        "aliases": ["grammar-guided decoding"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft narrower grammar-automaton subtype entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.state-sharding": {
        "aliases": ["ZeRO", "FSDP"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named sharded-optimizer/parameter implementations queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.paged-kv": {
        "aliases": ["PagedAttention"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft named paper/kernel entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.graph-replay": {
        "aliases": ["CUDA Graphs"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft vendor-specific graph capture entry term queued for pre-1.0 legacy alias audit.",
    },
    "mlsys.device-streams": {
        "aliases": ["CUDA streams"],
        "status": "pre-sidecar-legacy-queued-for-v1-audit",
        "reason": "Pre-sidecar 0.4-draft vendor-specific execution stream entry term queued for pre-1.0 legacy alias audit.",
    },
}
_REQUIRED_WITNESS_KEYS = ("compared_concept", "criterion_a", "criterion_b", "criterion_c", "scenario_a", "scenario_b")


def parse_taxonomy_semver(ver: str) -> tuple[int, int, int, int, int]:
    """Parse `<major>.<minor>[.<patch>][-(draft|rc<N>)]` into a monotonically comparable integer tuple."""
    m = _SEMVER_RE.match((ver or "").strip())
    if not m:
        raise ConceptGraphError(f"invalid taxonomy version string {ver!r}")
    major = int(m.group(1))
    minor = int(m.group(2))
    patch_v = int(m.group(3) or 0)
    pre = m.group(4) or ""
    if not pre:
        return (major, minor, patch_v, 2, 0)
    if pre == "draft":
        return (major, minor, patch_v, 0, 0)
    return (major, minor, patch_v, 1, int(m.group(5)))


def _norm_prov_tokens(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _reject_json_constant(val: str) -> None:
    raise ConceptGraphError(f"schema sync: non-finite JSON constant {val!r} is forbidden")


def _is_finite_num(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def canonical_taxonomy_json(doc: dict) -> str:
    """Serialize parsed taxonomy YAML into deterministic key-sorted JSON for semantic equality checks."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compute_non_guides_taxonomy_sha256(raw_tax: dict) -> str:
    """Compute canonical SHA-256 over all top-level ml-systems-taxonomy.yml sections except guides."""
    non_guides = {k: v for k, v in raw_tax.items() if k != "guides"}
    return hashlib.sha256(canonical_taxonomy_json(non_guides).encode("utf-8")).hexdigest()


def compute_campaign_entry_sha256(entry: dict) -> str:
    """Compute canonical SHA-256 over an applied_campaigns entry excluding campaign_entry_sha256."""
    body = {k: v for k, v in entry.items() if k != "campaign_entry_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=2)
def _cached_genesis_snapshot(gz_path_str: str) -> dict:
    gz_bytes = Path(gz_path_str).read_bytes()
    if hashlib.sha256(gz_bytes).hexdigest() != GENESIS_SNAPSHOT_GZ_SHA256:
        raise ConceptGraphError(
            "schema sync: genesis-0.4-baseline.json.gz SHA-256 does not match pinned 0.4-draft snapshot"
        )
    canon_bytes = gzip.decompress(gz_bytes)
    if hashlib.sha256(canon_bytes).hexdigest() != GENESIS_CANONICAL_JSON_SHA256:
        raise ConceptGraphError(
            "schema sync: decompressed genesis-0.4-baseline.json.gz SHA-256 does not match pinned 0.4-draft JSON"
        )
    return json.loads(canon_bytes.decode("utf-8"), parse_constant=_reject_json_constant)


def load_genesis_baseline_taxonomy(root: Path | None = None) -> dict:
    """Load a fresh deepcopy of the pinned 0.4-draft genesis taxonomy for deterministic replay."""
    cand = (root / GENESIS_SNAPSHOT_REL_PATH) if root is not None else None
    path = cand if (cand is not None and cand.exists()) else (REPO_ROOT_PATH / GENESIS_SNAPSHOT_REL_PATH)
    if not path.exists():
        raise ConceptGraphError(f"schema sync: missing genesis snapshot {GENESIS_SNAPSHOT_REL_PATH}")
    return deepcopy(_cached_genesis_snapshot(str(path)))


@lru_cache(maxsize=4)
def _cached_burned_clusters(burned_path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    return json.loads(Path(burned_path_str).read_text(encoding="utf-8"), parse_constant=_reject_json_constant)


def _load_burned_clusters(root: Path | None = None) -> dict:
    cand = (root / BURNED_CLUSTERS_REL_PATH) if root is not None else None
    path = cand if (cand is not None and cand.exists()) else (REPO_ROOT_PATH / BURNED_CLUSTERS_REL_PATH)
    if not path.exists():
        raise ConceptGraphError(f"schema sync: missing burned-clusters ledger {BURNED_CLUSTERS_REL_PATH}")
    bdoc = _cached_burned_clusters(str(path), path.stat().st_mtime_ns)
    if not isinstance(bdoc, dict) or bdoc.get("schema_version") != 3:
        raise ConceptGraphError(f"schema sync: {BURNED_CLUSTERS_REL_PATH} must be schema_version=3")
    hist = bdoc.get("historical_phase3_units")
    if not isinstance(hist, dict) or len(hist) != 382:
        raise ConceptGraphError(f"schema sync: {BURNED_CLUSTERS_REL_PATH} must pin all 382 historical_phase3_units")
    hist_sha = hashlib.sha256(canonical_taxonomy_json(hist).encode("utf-8")).hexdigest()
    if hist_sha != GENESIS_HISTORICAL_PHASE3_UNITS_SHA256 or bdoc.get("historical_phase3_units_sha256") != GENESIS_HISTORICAL_PHASE3_UNITS_SHA256:
        raise ConceptGraphError(f"schema sync: {BURNED_CLUSTERS_REL_PATH} historical_phase3_units_sha256 mismatch")
    init_counts = collections.Counter()
    for uid_k, umeta in hist.items():
        if not isinstance(umeta, dict):
            raise ConceptGraphError(f"schema sync: malformed historical_phase3_units[{uid_k!r}]")
        isp = umeta.get("initial_phase3_split")
        csp = umeta.get("current_exposure_split")
        if isp not in ("dev", "held_out_test", "reserve") or csp != "dev":
            raise ConceptGraphError(
                f"schema sync: historical_phase3_units[{uid_k!r}] must carry valid initial_phase3_split and current_exposure_split='dev'"
            )
        init_counts[isp] += 1
    if init_counts["dev"] != 200 or init_counts["held_out_test"] != 126 or init_counts["reserve"] != 56:
        raise ConceptGraphError(
            f"schema sync: historical_phase3_units initial_phase3_split counts {dict(init_counts)} disagree with pre-registered 200/126/56"
        )
    for k_arr in ("evaluation_attempts", "stage2_human_attempts"):
        if k_arr in bdoc and not isinstance(bdoc[k_arr], list):
            raise ConceptGraphError(f"schema sync: {BURNED_CLUSTERS_REL_PATH} {k_arr} must be a list")
    return bdoc


def canonical_patch_json_sha256(patch_obj: dict) -> str:
    """Compute canonical JSON SHA-256 of a parsed patch/final YAML mapping."""
    return hashlib.sha256(canonical_taxonomy_json(patch_obj).encode("utf-8")).hexdigest()


def compute_sidecar_content_sha256(prov_doc: dict) -> str:
    """Compute canonical SHA-256 over term-provenance.yml content excluding sidecar_content_sha256."""
    body = {k: v for k, v in prov_doc.items() if k != "sidecar_content_sha256"}
    return hashlib.sha256(canonical_taxonomy_json(body).encode("utf-8")).hexdigest()


def resolve_cluster_merges(raw_merges: dict[str, str], src_label: str = "burned-clusters.json") -> dict[str, str]:
    """Resolve multi-hop cluster_merges transitively with cycle detection."""
    clean = {str(k).strip(): str(v).strip() for k, v in (raw_merges or {}).items() if str(k).strip() and str(v).strip()}
    resolved: dict[str, str] = {}
    for k in clean:
        seen = {k}
        cur = clean[k]
        while cur in clean:
            if cur in seen:
                raise ConceptGraphError(f"schema sync: cycle in cluster_merges starting at {k!r} in {src_label}")
            seen.add(cur)
            cur = clean[cur]
        resolved[k] = cur
    return resolved


_LATIN_FOLD_TABLE = str.maketrans({
    "ß": "ss",
    "ø": "o",
    "đ": "d",
    "ł": "l",
})


def norm_person_or_org_token(raw: str, *, is_org: bool = False) -> str:
    """Normalize personal author or institutional organization names identically across all lineage checks."""
    s = (raw or "").lower().translate(_LATIN_FOLD_TABLE).replace("æ", "ae").replace("œ", "oe")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    toks = re.findall(r"[a-z0-9]+", s)
    if not toks:
        return ""
    if is_org:
        return f"inst:{' '.join(toks)}"
    if len(toks) >= 3:
        toks = [toks[0], *(t for t in toks[1:-1] if len(t) > 1), toks[-1]]
    return " ".join(toks)


def _extract_auth_and_org_sets(obj: dict) -> tuple[set[str], set[str]]:
    auths = {norm_person_or_org_token(str(a), is_org=False) for a in (obj.get("authors") or [])}
    orgs: set[str] = set()
    if obj.get("institutional_organization"):
        orgs.add(norm_person_or_org_token(str(obj["institutional_organization"]), is_org=True))
    for org in obj.get("institutional_organizations") or []:
        orgs.add(norm_person_or_org_token(str(org), is_org=True))
    auths.discard("")
    orgs.discard("")
    return auths, orgs


def compute_lineage_components(
    units: list[dict] | None,
    burned_doc: dict,
    *,
    include_all_exposed_meta: bool = True,
) -> dict[str, object]:
    """Unified transitive `<50%` personal-author and `>=1` institutional-organization overlap closure.

    Evaluates both cluster-level rosters and per-work `work_rosters` with separated personal-author
    and institutional-organization denominators so multi-work consortium clusters cannot dilute
    per-paper authorship overlap and institutional strings cannot skew author-count denominators.
    """
    merges = resolve_cluster_merges(burned_doc.get("cluster_merges") or {})
    internal_set = {merges.get(str(x).strip(), str(x).strip()) for x in (burned_doc.get("internal_clusters") or []) if str(x).strip()}
    internal_set.add("pf_internal_mlsysbook")
    editor_aff_set = {merges.get(str(x).strip(), str(x).strip()) for x in (burned_doc.get("editor_affiliated_clusters") or []) if str(x).strip()}
    exposed_meta = burned_doc.get("exposed_cluster_metadata") or {}

    records: list[tuple[str, set[str], set[str]]] = []
    present_ext_pfs: set[str] = set()
    has_internal = False
    ed_pfs: set[str] = set(editor_aff_set)

    def _append_work_rosters(pf_id: str, obj: dict) -> None:
        raw_wr = obj.get("work_rosters")
        wr_iter = raw_wr.values() if isinstance(raw_wr, dict) else (raw_wr if isinstance(raw_wr, list) else ())
        for wr in wr_iter:
            if isinstance(wr, dict):
                wa, wo = _extract_auth_and_org_sets(wr)
                if wa or wo:
                    records.append((pf_id, wa, wo))
            elif isinstance(wr, list):
                wa = {norm_person_or_org_token(str(a), is_org=False) for a in wr}
                wa.discard("")
                if wa:
                    records.append((pf_id, wa, set()))

    for u in units or []:
        raw_pf = str(u.get("provenance_family_id") or "").strip()
        if not raw_pf:
            continue
        pf = merges.get(raw_pf, raw_pf)
        is_int = (
            pf in internal_set
            or u.get("origin") == "internal"
            or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_"))
        )
        if is_int:
            has_internal = True
            internal_set.add(pf)
            continue
        present_ext_pfs.add(pf)
        if bool(u.get("editor_affiliated", False)) or bool(u.get("_editor_affiliated", False)):
            ed_pfs.add(pf)
        ua, uo = _extract_auth_and_org_sets(u)
        records.append((pf, ua, uo))
        _append_work_rosters(pf, u)

    for pf_raw, meta in sorted(exposed_meta.items()):
        if not isinstance(meta, dict):
            continue
        pf = merges.get(str(pf_raw).strip(), str(pf_raw).strip())
        if pf in internal_set or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_")):
            continue
        if not include_all_exposed_meta and pf not in present_ext_pfs:
            continue
        present_ext_pfs.add(pf)
        if bool(meta.get("editor_affiliated", False)):
            ed_pfs.add(pf)
        ma, mo = _extract_auth_and_org_sets(meta)
        records.append((pf, ma, mo))
        _append_work_rosters(pf, meta)

    ext_pfs = sorted(present_ext_pfs)
    parent = {p: p for p in ext_pfs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(records)):
        pf_i, auth_i, org_i = records[i]
        if not auth_i and not org_i:
            continue
        for j in range(i + 1, len(records)):
            pf_j, auth_j, org_j = records[j]
            if pf_i == pf_j:
                continue
            if org_i and org_j and (org_i & org_j):
                union(pf_i, pf_j)
                continue
            if auth_i and auth_j and (len(auth_i & auth_j) / min(len(auth_i), len(auth_j))) >= 0.5:
                union(pf_i, pf_j)

    pf_to_component: dict[str, str] = {}
    component_members: dict[str, list[str]] = {}
    if has_internal or internal_set:
        for ipf in sorted(internal_set):
            pf_to_component[ipf] = "eff_lineage:internal"
        if has_internal:
            component_members["eff_lineage:internal"] = sorted(internal_set)

    for p in ext_pfs:
        cid = f"eff_lineage:{find(p)}"
        pf_to_component[p] = cid
        component_members.setdefault(cid, []).append(p)

    comp_ed: set[str] = set()
    for p in ext_pfs:
        if p in ed_pfs:
            comp_ed.add(pf_to_component[p])

    return {
        "merges": merges,
        "internal_clusters": internal_set,
        "editor_affiliated_clusters": ed_pfs,
        "pf_to_component": pf_to_component,
        "component_members": component_members,
        "component_editor_affiliated": comp_ed,
    }


def _compute_warrant_lineages(warrant_sources: list[dict], burned_doc: dict) -> tuple[set[str], set[str]]:
    """Return `(all_effective_lineages, external_non_editor_effective_lineages)` under shared `<50%` overlap closure."""
    closure = compute_lineage_components(warrant_sources, burned_doc, include_all_exposed_meta=True)
    merges: dict[str, str] = closure["merges"]  # type: ignore[assignment]
    internal_set: set[str] = closure["internal_clusters"]  # type: ignore[assignment]
    pf_to_comp: dict[str, str] = closure["pf_to_component"]  # type: ignore[assignment]
    comp_ed: set[str] = closure["component_editor_affiliated"]  # type: ignore[assignment]

    all_lineages: set[str] = set()
    ext_non_ed_lineages: set[str] = set()
    for src in warrant_sources:
        raw_pf = str(src["provenance_family_id"]).strip()
        pf = merges.get(raw_pf, raw_pf)
        is_int = (
            pf in internal_set
            or src.get("origin") == "internal"
            or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_"))
        )
        if is_int:
            all_lineages.add("eff_lineage:internal")
        else:
            eff = pf_to_comp[pf]
            all_lineages.add(eff)
            if eff not in comp_ed and not bool(src.get("editor_affiliated", False)):
                ext_non_ed_lineages.add(eff)
    return all_lineages, ext_non_ed_lineages


def replay_patch_on_taxonomy_doc(doc: dict, change: dict, mode: str) -> dict:
    """ deterministically apply `change` to `doc` in place and return expected operation metadata."""
    if not isinstance(change, dict):
        raise ConceptGraphError("schema sync: replayed_patch must be a mapping")

    if mode == "guides-apply":
        g = doc.get("guides") or {}
        roles = {r["id"]: r for r in g.get("roles", [])}
        symptoms = {s["id"]: s for s in g.get("symptoms", [])}
        paths = {p["id"]: p for p in g.get("learning_paths", [])}
        expected_ops: list[str] = []

        def _g_remove(items: list[str], gone: list[str] | None, ctx: str) -> None:
            for x in gone or []:
                if x not in items:
                    raise ConceptGraphError(f"schema sync: guides replay {ctx}: cannot remove absent {x!r}")
                items.remove(x)

        def _g_add(items: list[str], new: list[str] | None) -> None:
            for x in new or []:
                if x not in items:
                    items.append(x)

        def _g_place(steps: list[dict], before: str | None, after: str | None) -> int:
            subs = [s["subject"] for s in steps]
            if before:
                if before not in subs:
                    raise ConceptGraphError(f"schema sync: guides replay before={before!r} not in steps")
                return subs.index(before)
            if after:
                if after not in subs:
                    raise ConceptGraphError(f"schema sync: guides replay after={after!r} not in steps")
                return subs.index(after) + 1
            return len(steps)

        for u in change.get("roles_update", []) or []:
            expected_ops.append(f"role_update:{u['id'].strip()}")
            r = roles[u["id"].strip()]
            _g_remove(r["core"], u.get("remove_core"), u["id"])
            _g_remove(r.setdefault("working", []), u.get("remove_working"), u["id"])
            _g_add(r["core"], u.get("add_core"))
            _g_add(r["working"], u.get("add_working"))
            r["working"] = [w for w in r["working"] if w not in r["core"]]
            if not r["working"]:
                r.pop("working", None)
            if u.get("summary"):
                r["summary"] = u["summary"].strip()
        for r in change.get("roles_add", []) or []:
            expected_ops.append(f"role_add:{r['id'].strip()}")
            new_r = deepcopy(r)
            g["roles"].append(new_r)
            roles[new_r["id"]] = new_r

        for u in change.get("symptoms_update", []) or []:
            expected_ops.append(f"symptom_update:{u['id'].strip()}")
            s = symptoms[u["id"].strip()]
            for key in ("causes", "checks", "levers", "quantities"):
                if f"remove_{key}" in u or f"add_{key}" in u:
                    _g_remove(s.setdefault(key, []), u.get(f"remove_{key}"), u["id"])
                    _g_add(s[key], u.get(f"add_{key}"))
                    if not s[key] and key in ("checks", "quantities"):
                        s.pop(key, None)
            for key in ("label", "signal"):
                if u.get(key):
                    s[key] = u[key].strip()
            if u.get("reorder"):
                lists = {k: s.get(k, []) for k in ("causes", "checks", "levers") if k in s}
                head = {k: v for k, v in s.items() if k not in ("causes", "checks", "levers")}
                s.clear()
                s.update(head)
                for key in u["reorder"]:
                    if key in lists:
                        s[key] = lists[key]
        for s in change.get("symptoms_add", []) or []:
            expected_ops.append(f"symptom_add:{s['id'].strip()}")
            new_s = deepcopy(s)
            g["symptoms"].append(new_s)
            symptoms[new_s["id"]] = new_s

        for u in change.get("paths_update", []) or []:
            expected_ops.append(f"path_update:{u['id'].strip()}")
            p = paths[u["id"].strip()]
            for mv in u.get("move", []) or []:
                matches = [st for st in p["steps"] if st["subject"] == mv["subject"]]
                step = matches[0]
                p["steps"].remove(step)
                p["steps"].insert(_g_place(p["steps"], mv.get("before"), mv.get("after")), step)
            for ins in u.get("insert", []) or []:
                p["steps"].insert(
                    _g_place(p["steps"], ins.get("before"), ins.get("after")),
                    {"subject": ins["subject"].strip(), "why": ins["why"].strip()},
                )
            for subject, why in (u.get("why") or {}).items():
                matches = [st for st in p["steps"] if st["subject"] == subject]
                matches[0]["why"] = why.strip()
        for p in change.get("paths_add", []) or []:
            expected_ops.append(f"path_add:{p['id'].strip()}")
            new_p = deepcopy(p)
            g["learning_paths"].append(new_p)
            paths[new_p["id"]] = new_p

        if change.get("version"):
            raise ConceptGraphError("schema sync: guides-apply is forbidden from modifying taxonomy version")
        if not str(doc.get("version") or "").endswith("-draft"):
            raise ConceptGraphError("schema sync: guides-apply may only mutate a -draft taxonomy")

        return {"expected_ops": sorted(expected_ops)}

    if mode == "finalize-structural-evaluation":
        if change != {"finalize_structural_evaluation": True}:
            raise ConceptGraphError(
                "schema sync: finalize-structural-evaluation replayed_patch must be {'finalize_structural_evaluation': True}"
            )
        return {"expected_ops": ["finalize_evaluation"]}

    rows = {row["id"]: row for area in doc["areas"] for row in area["rows"]}
    code2row = {row["code"]: row["id"] for row in rows.values() if "code" in row}
    baseline_row_sec = {rid: list(row.get("secondary_concepts", []) or []) for rid, row in rows.items()}
    concepts = {c["id"]: c for c in doc["concepts"]}
    families = {f["id"]: f for f in doc["families"]}
    relations = {r["id"]: r for r in doc["relations"]}

    def resolve_row_id(home_ref: str) -> str:
        return code2row.get(home_ref, home_ref)

    def family_for(home_ref: str, cid: str) -> str:
        rid = resolve_row_id(home_ref)
        members = [c["family"] for c in doc["concepts"] if c.get("home") == rid and "family" in c]
        if not members:
            raise ConceptGraphError(f"schema sync: replay concept {cid} in empty subject {rid} lacks explicit family")
        return collections.Counter(members).most_common(1)[0][0]

    expected_ops = []
    exp_c_added: list[str] = []
    exp_c_retired: list[str] = []
    exp_defs_updated: list[str] = []
    exp_aliases_added: dict[str, list[str]] = {}
    exp_aliases_attested: dict[str, list[str]] = {}
    exp_aliases_removed: dict[str, list[str]] = {}
    exp_other_c_upd: list[str] = []
    exp_sec_added: dict[str, list[str]] = {}
    exp_sec_removed: dict[str, list[str]] = {}
    exp_other_s_upd: list[str] = []
    adopted_alias_norms_by_cid: dict[str, set[str]] = collections.defaultdict(set)

    for u in change.get("concepts_update", []) or []:
        cid = u["id"].strip()
        expected_ops.append(f"update:{cid}")
        node = concepts[cid]
        for key in ("label", "definition"):
            if u.get(key):
                node[key] = " ".join(str(u[key]).split())
        if u.get("scope"):
            node["scope"] = u["scope"].strip()
        if u.get("definition"):
            exp_defs_updated.append(cid)
        if u.get("label") or u.get("scope"):
            exp_other_c_upd.append(cid)
        rem_list = [str(x).strip() for x in u.get("remove_aliases", []) or []]
        rem_norms = {_norm_prov_tokens(x) for x in rem_list}
        base_aliases = [str(a) for a in node.get("aliases", []) or []]
        if rem_list:
            exp_aliases_removed[cid] = rem_list
        aliases = [a for a in base_aliases if _norm_prov_tokens(a) not in rem_norms]
        added_alist: list[str] = []
        for a in u.get("add_aliases", []) or []:
            astr = str(a).strip()
            aliases.append(astr)
            added_alist.append(astr)
            adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(astr))
        if added_alist:
            exp_aliases_added[cid] = added_alist
        attested_alist: list[str] = []
        for a in u.get("attest_existing_aliases", []) or []:
            astr = str(a).strip()
            if astr not in aliases or astr in added_alist:
                raise ConceptGraphError(
                    f"schema sync: attest_existing_aliases {astr!r} on {cid} must name an already-active surviving alias"
                )
            attested_alist.append(astr)
        if attested_alist:
            exp_aliases_attested[cid] = attested_alist
        if u.get("label"):
            adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(node["label"]))
        saved_added_in = node.pop("added_in", None)
        if aliases:
            node["aliases"] = aliases
        else:
            node.pop("aliases", None)
        if saved_added_in is not None:
            node["added_in"] = saved_added_in

    for mv in change.get("moves", []) or []:
        mcid = mv["id"].strip()
        expected_ops.append(f"move:{mcid}")
        mhome = resolve_row_id(mv["home"].strip())
        concepts[mcid]["home"] = mhome
        dest_row = rows[mhome]
        if "secondary_concepts" in dest_row and mcid in dest_row["secondary_concepts"]:
            cleaned_sec = [s for s in dest_row["secondary_concepts"] if s != mcid]
            if cleaned_sec:
                dest_row["secondary_concepts"] = cleaned_sec
            else:
                dest_row.pop("secondary_concepts", None)
            if mcid not in exp_sec_removed.setdefault(mhome, []):
                exp_sec_removed[mhome].append(mcid)

    target_version = str(change.get("version") or doc["version"]).strip()
    for c in change.get("concepts_add", []) or []:
        cid = c["id"].strip()
        expected_ops.append(f"add:{cid}")
        home_rid = resolve_row_id(c["home"].strip())
        node = {
            "id": cid,
            "label": " ".join(str(c["label"]).split()),
            "home": home_rid,
            "family": (c.get("family") or family_for(c["home"], cid)).strip(),
            "definition": " ".join(str(c["definition"]).split()),
            "status": "draft",
        }
        adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(node["label"]))
        if c.get("aliases"):
            node["aliases"] = [str(x).strip() for x in c["aliases"]]
            exp_aliases_added[cid] = list(node["aliases"])
            for ca in node["aliases"]:
                adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(ca))
        if c.get("scope"):
            node["scope"] = str(c["scope"]).strip()
        node["added_in"] = target_version
        exp_c_added.append(cid)
        doc["concepts"].append(node)
        concepts[cid] = node

    migrate_hv_map: dict[str, str] = {}
    for r in change.get("concepts_retire", []) or []:
        rcid = r["id"].strip()
        expected_ops.append(f"retire:{rcid}")
        rep = r["replaced_by"].strip()
        former = concepts.pop(rcid)
        exp_c_retired.append(rcid)
        doc["concepts"] = [c for c in doc["concepts"] if c["id"] != rcid]
        doc.setdefault("retired", [])
        former_lbl = str(former.get("label") or "").strip()
        former_als = [str(a) for a in (former.get("aliases") or [])]
        ret_entry: dict[str, object] = {"id": rcid}
        if former_lbl:
            ret_entry["label"] = former_lbl
        if former_als:
            ret_entry["aliases"] = list(former_als)
        ret_entry["replaced_by"] = rep
        ret_entry["version"] = str(r.get("version") or target_version).strip()
        ret_entry["reason"] = " ".join(str(r["reason"]).split())
        doc["retired"].append(ret_entry)
        doc["placement_examples"] = [x for x in doc.get("placement_examples", []) if x["concept"] != rcid]
        for rid_k, row_obj in rows.items():
            if "secondary_concepts" in row_obj and rcid in row_obj["secondary_concepts"]:
                sec = [s for s in row_obj["secondary_concepts"] if s != rcid]
                if sec:
                    row_obj["secondary_concepts"] = sec
                else:
                    row_obj.pop("secondary_concepts", None)
                if rcid not in exp_sec_removed.setdefault(rid_k, []):
                    exp_sec_removed[rid_k].append(rcid)
        if r.get("migrate_hierarchy_and_views") is True:
            migrate_hv_map[rcid] = rep

    if doc.get("retired"):
        rep_map = {rec_r["id"]: rec_r["replaced_by"] for rec_r in doc["retired"]}
        for rec_r in doc["retired"]:
            cur = rec_r["replaced_by"]
            seen_chain = {rec_r["id"]}
            while cur in rep_map:
                if cur in seen_chain:
                    raise ConceptGraphError(f"schema sync: cycle in retired replaced_by chain at {rec_r['id']}")
                seen_chain.add(cur)
                cur = rep_map[cur]
            rec_r["replaced_by"] = cur
            if rec_r["id"] in migrate_hv_map:
                migrate_hv_map[rec_r["id"]] = cur
            active_rep_norms = (
                adopted_alias_norms_by_cid.get(cur, set())
                | {_norm_prov_tokens(str(concepts[cur].get("label") or ""))}
                | {_norm_prov_tokens(str(a)) for a in (concepts[cur].get("aliases") or [])}
            )
            if rec_r.get("label") and _norm_prov_tokens(str(rec_r["label"])) in active_rep_norms:
                rec_r.pop("label", None)
            if rec_r.get("aliases"):
                rem_ret_a = [a for a in rec_r["aliases"] if _norm_prov_tokens(a) not in active_rep_norms]
                if rem_ret_a:
                    rec_r["aliases"] = rem_ret_a
                else:
                    rec_r.pop("aliases", None)

    if migrate_hv_map:
        sh = doc.get("skos_hierarchy")
        if isinstance(sh, dict):
            if isinstance(sh.get("hierarchy"), list):
                new_h = []
                seen_h: set[tuple[str, str]] = set()
                for h_item in sh["hierarchy"]:
                    nw = migrate_hv_map.get(h_item["narrower"], h_item["narrower"])
                    br = migrate_hv_map.get(h_item["broader"], h_item["broader"])
                    if nw == br or (nw, br) in seen_h:
                        continue
                    seen_h.add((nw, br))
                    upd_h = dict(h_item)
                    upd_h["narrower"] = nw
                    upd_h["broader"] = br
                    new_h.append(upd_h)
                sh["hierarchy"] = new_h
            if isinstance(sh.get("top_concepts"), list):
                new_tc = []
                for tc in sh["top_concepts"]:
                    mtc = migrate_hv_map.get(tc, tc)
                    if mtc not in new_tc:
                        new_tc.append(mtc)
                sh["top_concepts"] = new_tc
        for vw in doc.get("views") or []:
            if isinstance(vw, dict) and isinstance(vw.get("concepts"), list):
                new_vc = []
                for vc in vw["concepts"]:
                    mvc = migrate_hv_map.get(vc, vc)
                    if mvc not in new_vc:
                        new_vc.append(mvc)
                vw["concepts"] = new_vc

    removed_edges: set[str] = set()
    for x in change.get("relationships_remove", []) or []:
        sx = x.strip()
        expected_ops.append(f"rel_remove:{sx}")
        removed_edges.add(sx)
    doc["relationships"] = [e for e in doc["relationships"] if e["id"] not in removed_edges]
    edges_by_id = {e["id"]: e for e in doc["relationships"]}
    for u in change.get("relationships_update", []) or []:
        uid = u["id"].strip()
        expected_ops.append(f"rel_update:{uid}")
        edge = edges_by_id[uid]
        edge.update({k: v for k, v in u.items() if k in ("predicate", "conditions", "references", "reference_role")})
        if "relation" in u:
            rel_v = u["relation"].strip()
            edge["relation"] = rel_v
            edge["kind"] = relations[rel_v]["kind"]

    edge_ids = set(edges_by_id)
    for r in change.get("relationships_add", []) or []:
        src, tgt = r["source"].strip(), r["target"].strip()
        expected_ops.append(f"edge:{min(src, tgt)}--{max(src, tgt)}")
        rel_v = r["relation"].strip()
        base = f"rel.{src.removeprefix('mlsys.')}.{tgt.removeprefix('mlsys.')}"
        eid, n = base, 2
        while eid in edge_ids:
            eid, n = f"{base}.{n}", n + 1
        edge_ids.add(eid)
        edge = {
            "id": eid,
            "source": src,
            "target": tgt,
            "predicate": r["predicate"].strip(),
            "conditions": r["conditions"].strip(),
            "references": list(r.get("references") or families[concepts[src]["family"]]["references"]),
            "basis": "editorial-synthesis",
            "status": "draft",
            "kind": relations[rel_v]["kind"],
            "reference_role": "reading-basis-not-entailment-proof",
            "relation": rel_v,
        }
        doc["relationships"].append(edge)

    gaps = {g["id"]: g for g in doc.get("gaps", []) or []}
    for gid in change.get("gaps_remove", []) or []:
        sgid = gid.strip()
        expected_ops.append(f"gap_remove:{sgid}")
        gaps.pop(sgid)
    for g in change.get("gaps_update", []) or []:
        ugid = g["id"].strip()
        expected_ops.append(f"gap_update:{ugid}")
        for k, v in g.items():
            if k == "id":
                continue
            gaps[ugid][k] = resolve_row_id(v.strip()) if k == "subject_row" else v
    for g in change.get("gaps_add", []) or []:
        agid = g["id"].strip()
        expected_ops.append(f"gap_add:{agid}")
        gaps[agid] = {
            "id": agid,
            "subject_row": resolve_row_id(g["subject_row"].strip()),
            "topic": g["topic"].strip(),
            "status": "candidate",
            "reason": g["reason"].strip(),
        }
    doc["gaps"] = list(gaps.values())

    for su in change.get("subject_updates", []) or []:
        rid = resolve_row_id(su["row"].strip())
        expected_ops.append(f"row:{rid}")
        row = rows[rid]
        if "remove_includes" in su or "add_includes" in su:
            exp_other_s_upd.append(rid)
            base_inc = list(row.get("includes", []) or [])
            row["includes"] = [i for i in base_inc if i not in set(su.get("remove_includes", []) or [])] + list(
                su.get("add_includes", []) or []
            )
        if "remove_secondary_concepts" in su or "add_secondary_concepts" in su:
            orig_sec = baseline_row_sec[rid]
            cur_sec = list(row.get("secondary_concepts", []) or [])
            rem_secs = [s.strip() for s in su.get("remove_secondary_concepts", []) or []]
            for rs in rem_secs:
                if rs in orig_sec and rs not in exp_sec_removed.setdefault(rid, []):
                    exp_sec_removed[rid].append(rs)
            sec = [s for s in cur_sec if s not in set(rem_secs)]
            added_secs_here: list[str] = []
            for raw_s in su.get("add_secondary_concepts", []) or []:
                s = raw_s.strip()
                if s not in sec:
                    sec.append(s)
                added_secs_here.append(s)
            if added_secs_here:
                exp_sec_added[rid] = added_secs_here
            if sec:
                row["secondary_concepts"] = sec
            else:
                row.pop("secondary_concepts", None)

    for pe in change.get("placement_examples_add", []) or []:
        pcid = pe["concept"].strip()
        expected_ops.append(f"pe:{pcid}")
        phome = resolve_row_id(pe["home"].strip())
        doc.setdefault("placement_examples", [])
        doc["placement_examples"] = [x for x in doc["placement_examples"] if x["concept"] != pcid]
        doc["placement_examples"].append({"concept": pcid, "home": phome, "why": pe["why"].strip()})

    exp_guides_replaced = False
    if change.get("guides_replace") is not None:
        if not isinstance(change["guides_replace"], dict):
            raise ConceptGraphError("schema sync: guides_replace must be a mapping")
        expected_ops.append("guides:replace")
        doc["guides"] = deepcopy(change["guides_replace"])
        exp_guides_replaced = True

    if change.get("version"):
        expected_ops.append("version")
        doc["version"] = str(change["version"]).strip()

    return {
        "expected_ops": sorted(expected_ops),
        "concepts_added": sorted(exp_c_added),
        "concepts_retired": sorted(exp_c_retired),
        "definitions_updated": sorted(exp_defs_updated),
        "aliases_added": {k: list(v) for k, v in sorted(exp_aliases_added.items())},
        "aliases_attested": {k: list(v) for k, v in sorted(exp_aliases_attested.items())},
        "aliases_removed": {k: list(v) for k, v in sorted(exp_aliases_removed.items())},
        "other_concept_updates": sorted(exp_other_c_upd),
        "secondary_concepts_added": {k: list(v) for k, v in sorted(exp_sec_added.items())},
        "secondary_concepts_removed": {k: list(v) for k, v in sorted(exp_sec_removed.items())},
        "other_subject_updates": sorted(exp_other_s_upd),
        "guides_replaced": exp_guides_replaced,
    }


def init_genesis_provenance_state(genesis_doc: dict) -> dict[str, object]:
    """Initialize top-level term-provenance state at 0.4-draft genesis prior to applied_campaigns[0]."""
    concepts_by_id = {c["id"]: c for c in genesis_doc.get("concepts", [])}
    return {
        "concepts": {},
        "subjects": {},
        "retired_concepts": {},
        "grandfathered_draft_alias_exceptions": {
            gcid: deepcopy(grec)
            for gcid, grec in BOOTSTRAP_GRANDFATHERED_ALIAS_EXCEPTIONS.items()
            if gcid in concepts_by_id
            and set(grec["aliases"]).issubset(set(concepts_by_id[gcid].get("aliases") or []))
        },
        "pending_campaign_ids": [],
        "pending_pre_evaluation_override": False,
        "last_reconcile_receipt": None,
        "release_governance_receipt": None,
    }


_STAGE2_PAIR_ORDER: tuple[str, ...] = ("F", "S", "P", "Q", "T1", "T2")
_STAGE2_ANCHOR_ROW_QUOTAS: dict[str, dict[str, int]] = {
    "Q": {"F": 8, "S": 7, "P": 7, "T1": 7, "T2": 7},
    "P": {"F": 6, "S": 6, "Q": 6, "T1": 6, "T2": 6},
    "S": {"F": 4, "P": 5, "Q": 5, "T1": 5, "T2": 5},
    "F": {"S": 5, "P": 5, "Q": 5, "T1": 1, "T2": 2},
    "T1": {"F": 1, "S": 1, "P": 2, "Q": 2, "T2": 0},
    "T2": {"F": 1, "S": 1, "P": 1, "Q": 2, "T1": 1},
}
_SPECIALIST_SLOTS_EXPECTED = frozenset({
    "F3", "S4", "P1", "P2", "P3", "P5", "Q1", "Q2", "Q3", "Q4a", "Q4b", "Q5", "Q6",
})
_CHAIR_SIGN_OFF_AREAS_TO_SEAT: dict[str, str] = {
    "F1": "F",
    "F2": "F",
    "S1": "S",
    "S2": "S",
    "S3": "S",
    "P4": "P",
    "T1": "T",
    "T2": "T",
}
_BOARD_SEATS_EXPECTED = frozenset({"F", "S", "P", "Q", "T", "V1", "V2"})


def expected_stage2_anchor_schedule() -> dict[str, list[tuple[str, str, str]]]:
    """Deterministically generate the 40-anchor 3-receiver secondary-pair schedule from candidate-experts.md §4.2."""
    rank = {p: idx for idx, p in enumerate(_STAGE2_PAIR_ORDER)}
    counts_by_group = {"Q": 12, "P": 10, "S": 8, "F": 6, "T1": 2, "T2": 2}
    schedule: dict[str, list[tuple[str, str, str]]] = {}
    for g, n_anc in counts_by_group.items():
        rem = dict(_STAGE2_ANCHOR_ROW_QUOTAS[g])
        group_rows: list[tuple[str, str, str]] = []
        for _ in range(n_anc):
            cands = sorted(rem.keys(), key=lambda p: (-rem[p], rank[p]))
            picked = sorted(cands[:3], key=lambda p: rank[p])
            for p in picked:
                rem[p] -= 1
            group_rows.append((picked[0], picked[1], picked[2]))
        if any(v != 0 for v in rem.values()):
            raise ConceptGraphError(f"schema sync: internal anchor schedule quota residual for {g}: {rem}")
        schedule[g] = group_rows
    return schedule


def _masi_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    jacc = inter / len(a | b)
    if a == b:
        mono = 1.0
    elif a.issubset(b) or b.issubset(a):
        mono = 2.0 / 3.0
    else:
        mono = 1.0 / 3.0
    return jacc * mono


def _clamp_ieee_signed_unit(val: float) -> float:
    if 1.0 < val <= 1.0 + 1e-12:
        return 1.0
    if -1.0 - 1e-12 <= val < -1.0:
        return -1.0
    return val


def _clamp_ieee_unit(val: float) -> float:
    if 1.0 < val <= 1.0 + 1e-12:
        return 1.0
    if -1e-12 <= val < 0.0:
        return 0.0
    return val


def _weighted_krippendorff_nominal_alpha(items_vals: list[list[str]], weights: list[float]) -> tuple[float, bool]:
    n_val = 0.0
    d_o_num = 0.0
    marginals: dict[str, float] = {}
    for vals, w in zip(items_vals, weights):
        if w <= 0.0:
            continue
        r_i = len(vals)
        if r_i < 2:
            raise ConceptGraphError("schema sync: Krippendorff alpha requires r_i >= 2 raters per item")
        n_val += w * r_i
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        for v, c in counts.items():
            marginals[v] = marginals.get(v, 0.0) + w * c
        same_pairs = sum(c * (c - 1) for c in counts.values())
        diff_pairs = r_i * (r_i - 1) - same_pairs
        d_o_num += w * (diff_pairs / (r_i - 1))
    if n_val <= 1.0:
        return (1.0, True)
    d_o = d_o_num / n_val
    d_e_num = n_val * n_val - sum(nc * nc for nc in marginals.values())
    d_e = d_e_num / (n_val * (n_val - 1.0))
    if d_e <= 1e-15:
        return ((1.0 if d_o <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit(1.0 - d_o / d_e), False)


def _weighted_krippendorff_masi_alpha(
    items_sets: list[list[frozenset[str]]],
    weights: list[float],
) -> tuple[float, bool]:
    distinct_sets: list[frozenset[str]] = []
    set_to_idx: dict[frozenset[str], int] = {}
    encoded_items: list[list[int]] = []
    for s_list in items_sets:
        row_ids: list[int] = []
        for fs in s_list:
            idx = set_to_idx.get(fs)
            if idx is None:
                idx = len(distinct_sets)
                set_to_idx[fs] = idx
                distinct_sets.append(fs)
            row_ids.append(idx)
        encoded_items.append(row_ids)
    n_pat = len(distinct_sets)
    sim_lookup: dict[tuple[int, int], float] = {}
    nonzero_pairs: list[tuple[int, int, float]] = []
    for p in range(n_pat):
        sim_lookup[(p, p)] = 1.0
        for q in range(p + 1, n_pat):
            sim = _masi_similarity(distinct_sets[p], distinct_sets[q])
            if sim > 0.0:
                sim_lookup[(p, q)] = sim
                sim_lookup[(q, p)] = sim
                nonzero_pairs.append((p, q, sim))

    item_disagreements: list[float] = []
    for row_ids in encoded_items:
        r_i = len(row_ids)
        if r_i < 2:
            raise ConceptGraphError("schema sync: MASI alpha requires r_i >= 2 raters per item")
        dis_sum = 0.0
        for u in range(r_i):
            pu = row_ids[u]
            for v in range(u + 1, r_i):
                pv = row_ids[v]
                if pu != pv:
                    dis_sum += 2.0 * (1.0 - sim_lookup.get((pu, pv), 0.0))
        item_disagreements.append(dis_sum / (r_i - 1))

    return _fast_masi_alpha_from_precomputed(encoded_items, item_disagreements, nonzero_pairs, n_pat, weights)


def _fast_masi_alpha_from_precomputed(
    encoded_items: list[list[int]],
    item_disagreements: list[float],
    nonzero_pairs: list[tuple[int, int, float]],
    n_pat: int,
    weights: list[float],
) -> tuple[float, bool]:
    n_val = 0.0
    d_o_num = 0.0
    n_p = [0.0] * n_pat
    for row_ids, dis_i, w in zip(encoded_items, item_disagreements, weights):
        if w <= 0.0:
            continue
        n_val += w * len(row_ids)
        d_o_num += w * dis_i
        for pid in row_ids:
            n_p[pid] += w
    if n_val <= 1.0:
        return (1.0, True)
    d_o = d_o_num / n_val
    off_diag_mass = n_val * n_val - sum(x * x for x in n_p)
    pos_sim_mass = 0.0
    for p, q, sim in nonzero_pairs:
        np_v = n_p[p]
        if np_v > 0.0:
            nq_v = n_p[q]
            if nq_v > 0.0:
                pos_sim_mass += np_v * nq_v * sim
    d_e_num = off_diag_mass - 2.0 * pos_sim_mass
    if d_e_num < 0.0 and d_e_num > -1e-11:
        d_e_num = 0.0
    d_e = d_e_num / (n_val * (n_val - 1.0))
    if d_e <= 1e-15:
        return ((1.0 if d_o <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit(1.0 - d_o / d_e), False)


def _weighted_pair_stratified_cohen_kappa(
    groups: list[str],
    subj1: list[str],
    subj2: list[str],
    weights: list[float],
) -> tuple[float, bool]:
    total_w = sum(w for w in weights if w > 0.0)
    if total_w <= 0.0:
        return (1.0, True)
    by_g: dict[str, list[int]] = collections.defaultdict(list)
    for idx, g in enumerate(groups):
        if weights[idx] > 0.0:
            by_g[g].append(idx)
    weighted_kappa_sum = 0.0
    any_zero_var = False
    for g, idxs in sorted(by_g.items()):
        n_g = sum(weights[i] for i in idxs)
        if n_g <= 0.0:
            continue
        po_num = 0.0
        m1: dict[str, float] = {}
        m2: dict[str, float] = {}
        for i in idxs:
            w = weights[i]
            s1, s2 = subj1[i], subj2[i]
            if s1 == s2:
                po_num += w
            m1[s1] = m1.get(s1, 0.0) + w
            m2[s2] = m2.get(s2, 0.0) + w
        po_g = po_num / n_g
        pe_g = sum((v1 / n_g) * (m2.get(s, 0.0) / n_g) for s, v1 in m1.items())
        if 1.0 - pe_g <= 1e-15:
            any_zero_var = True
            k_g = 1.0 if abs(po_g - 1.0) <= 1e-15 else 0.0
        else:
            k_g = _clamp_ieee_signed_unit((po_g - pe_g) / (1.0 - pe_g))
        weighted_kappa_sum += (n_g / total_w) * k_g
    return (_clamp_ieee_signed_unit(weighted_kappa_sum), any_zero_var)


def _weighted_hooper_rolling(
    sets1: list[frozenset[str]],
    sets2: list[frozenset[str]],
    weights: list[float],
) -> dict[str, float]:
    tot_w = 0.0
    cond_w = 0.0
    cond_h = 0.0
    cond_r = 0.0
    all_h = 0.0
    all_r = 0.0
    empty_w = 0.0
    for a, b, w in zip(sets1, sets2, weights):
        if w <= 0.0:
            continue
        tot_w += w
        if a and b:
            inter = len(a & b)
            hi = inter / len(a | b)
            ri = (2.0 * inter) / (len(a) + len(b))
            cond_w += w
            cond_h += w * hi
            cond_r += w * ri
            all_h += w * hi
            all_r += w * ri
        elif not a and not b:
            empty_w += w
            all_h += w * 1.0
            all_r += w * 1.0
        else:
            empty_w += w
    if tot_w <= 0.0 or cond_w <= 0.0:
        raise ConceptGraphError("schema sync: degenerate zero denominator in Hooper/Rolling overlap computation")
    return {
        "cond_hooper": _clamp_ieee_unit(cond_h / cond_w),
        "all_hooper": _clamp_ieee_unit(all_h / tot_w),
        "cond_rolling": _clamp_ieee_unit(cond_r / cond_w),
        "all_rolling": _clamp_ieee_unit(all_r / tot_w),
        "empty_rate": _clamp_ieee_unit(empty_w / tot_w),
    }


def _weighted_fleiss_kappa_8rater(anchor_subj_lists: list[list[str]], weights: list[float]) -> tuple[float, bool]:
    w_sum = sum(w for w in weights if w > 0.0)
    if w_sum <= 0.0:
        return (1.0, True)
    po_acc = 0.0
    cat_totals: dict[str, float] = {}
    for subjs, w in zip(anchor_subj_lists, weights):
        if w <= 0.0:
            continue
        if len(subjs) != 8:
            raise ConceptGraphError("schema sync: 8-rater Fleiss kappa requires 8 subject assignments per anchor")
        counts: dict[str, int] = {}
        for s in subjs:
            counts[s] = counts.get(s, 0) + 1
        p_i = sum(c * (c - 1) for c in counts.values()) / 56.0
        po_acc += w * p_i
        for s, c in counts.items():
            cat_totals[s] = cat_totals.get(s, 0.0) + w * c
    p_o = po_acc / w_sum
    denom_8n = 8.0 * w_sum
    p_e = sum((tot / denom_8n) ** 2 for tot in cat_totals.values())
    if 1.0 - p_e <= 1e-15:
        return ((1.0 if abs(p_o - 1.0) <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit((p_o - p_e) / (1.0 - p_e)), False)


@lru_cache(maxsize=1)
def run_stage2_estimator_self_tests() -> bool:
    """Execute the 4 mandatory unit-test fixtures from candidate-experts.md §4.3 before Stage 2 evaluation."""
    groups = ["F"] * 6 + ["Q"] * 6
    s1 = ["F1.1", "F1.1", "F1.2", "F2.1", "F2.1", "F3.1", "Q1.1", "Q1.1", "Q2.1", "Q4.1", "Q4.1", "Q4.2"]
    s2 = ["F1.1", "F1.2", "F1.2", "F2.1", "F3.1", "F3.1", "Q1.1", "Q2.1", "Q2.1", "Q4.1", "Q4.2", "Q4.2"]
    c1 = [
        frozenset({"c1", "c2"}),
        frozenset({"c1"}),
        frozenset(),
        frozenset({"c3"}),
        frozenset({"c2", "c3"}),
        frozenset({"c4"}),
        frozenset({"q1", "q2"}),
        frozenset(),
        frozenset({"q2"}),
        frozenset({"q3", "q4"}),
        frozenset({"q3"}),
        frozenset({"q5"}),
    ]
    c2 = [
        frozenset({"c1", "c2"}),
        frozenset({"c1", "c2"}),
        frozenset(),
        frozenset(),
        frozenset({"c3", "c4"}),
        frozenset({"c4"}),
        frozenset({"q1"}),
        frozenset({"q1"}),
        frozenset({"q2"}),
        frozenset({"q3", "q4"}),
        frozenset({"q4"}),
        frozenset({"q5"}),
    ]
    w_uni = [1.0] * 12

    # Fixture 1: Orientation-swap invariance across arbitrary item subset & 8-rater anchor permutations.
    a_sub_base, _ = _weighted_krippendorff_nominal_alpha([[a, b] for a, b in zip(s1, s2)], w_uni)
    a_masi_base, _ = _weighted_krippendorff_masi_alpha([[a, b] for a, b in zip(c1, c2)], w_uni)
    hr_base = _weighted_hooper_rolling(c1, c2, w_uni)
    swap_idx = {0, 1, 3, 4, 7, 10}
    s1_sw = [s2[i] if i in swap_idx else s1[i] for i in range(12)]
    s2_sw = [s1[i] if i in swap_idx else s2[i] for i in range(12)]
    c1_sw = [c2[i] if i in swap_idx else c1[i] for i in range(12)]
    c2_sw = [c1[i] if i in swap_idx else c2[i] for i in range(12)]
    a_sub_sw, _ = _weighted_krippendorff_nominal_alpha([[a, b] for a, b in zip(s1_sw, s2_sw)], w_uni)
    a_masi_sw, _ = _weighted_krippendorff_masi_alpha([[a, b] for a, b in zip(c1_sw, c2_sw)], w_uni)
    hr_sw = _weighted_hooper_rolling(c1_sw, c2_sw, w_uni)
    anc_8 = [["F1.1"] * 5 + ["F1.2"] * 3, ["Q4.1"] * 6 + ["Q4.2"] * 2, ["S1.1"] * 4 + ["S1.2"] * 4]
    anc_8_perm = [list(reversed(r)) for r in anc_8]
    fk_base, _ = _weighted_fleiss_kappa_8rater(anc_8, [1.0, 1.0, 1.0])
    fk_sw, _ = _weighted_fleiss_kappa_8rater(anc_8_perm, [1.0, 1.0, 1.0])
    if (
        abs(a_sub_base - a_sub_sw) >= 1e-12
        or abs(a_masi_base - a_masi_sw) >= 1e-12
        or abs(hr_base["cond_hooper"] - hr_sw["cond_hooper"]) >= 1e-12
        or abs(hr_base["all_hooper"] - hr_sw["all_hooper"]) >= 1e-12
        or abs(hr_base["cond_rolling"] - hr_sw["cond_rolling"]) >= 1e-12
        or abs(hr_base["all_rolling"] - hr_sw["all_rolling"]) >= 1e-12
        or abs(fk_base - fk_sw) >= 1e-12
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 1 (orientation-swap invariance) failed")

    # Fixture 2: Uniform-weight identity against unweighted direct formulas.
    n_items = len(s1)
    n_val_unw = 2 * n_items
    do_sub_unw = sum(1.0 if a != b else 0.0 for a, b in zip(s1, s2)) / n_items
    cnt_sub = collections.Counter(s1 + s2)
    de_sub_unw = sum(cnt_sub[x] * cnt_sub[y] for x in cnt_sub for y in cnt_sub if x != y) / (n_val_unw * (n_val_unw - 1))
    alpha_sub_unw = 1.0 - do_sub_unw / de_sub_unw
    do_masi_unw = sum(1.0 - _masi_similarity(a, b) for a, b in zip(c1, c2)) / n_items
    all_sets_flat = c1 + c2
    de_masi_unw = sum(
        1.0 - _masi_similarity(all_sets_flat[u], all_sets_flat[v])
        for u in range(n_val_unw)
        for v in range(n_val_unw)
        if u != v
    ) / (n_val_unw * (n_val_unw - 1))
    alpha_masi_unw = 1.0 - do_masi_unw / de_masi_unw
    kb_weighted, _ = _weighted_pair_stratified_cohen_kappa(groups, s1, s2, w_uni)
    kb_unw = 0.0
    for g_id in ("F", "Q"):
        g_idx = [i for i, gg in enumerate(groups) if gg == g_id]
        po = sum(1 for i in g_idx if s1[i] == s2[i]) / len(g_idx)
        c_g1 = collections.Counter(s1[i] for i in g_idx)
        c_g2 = collections.Counter(s2[i] for i in g_idx)
        pe = sum((c_g1[k] / len(g_idx)) * (c_g2[k] / len(g_idx)) for k in set(c_g1) | set(c_g2))
        kb_unw += (len(g_idx) / n_items) * ((po - pe) / (1.0 - pe))
    if (
        abs(a_sub_base - alpha_sub_unw) >= 1e-12
        or abs(a_masi_base - alpha_masi_unw) >= 1e-12
        or abs(kb_weighted - kb_unw) >= 1e-12
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 2 (uniform-weight identity) failed")

    # Fixture 3: Integer cluster-replication equivalence (sum k_i = N).
    int_mults = [2, 0, 3, 1, 2, 0, 1, 0, 1, 1, 1, 0]  # sum == 12
    a_sub_int, _ = _weighted_krippendorff_nominal_alpha(
        [[a, b] for a, b in zip(s1, s2)],
        [float(k) for k in int_mults],
    )
    a_masi_int, _ = _weighted_krippendorff_masi_alpha(
        [[a, b] for a, b in zip(c1, c2)],
        [float(k) for k in int_mults],
    )
    dup_sub = [[s1[i], s2[i]] for i, k in enumerate(int_mults) for _ in range(k)]
    dup_masi = [[c1[i], c2[i]] for i, k in enumerate(int_mults) for _ in range(k)]
    a_sub_dup, _ = _weighted_krippendorff_nominal_alpha(dup_sub, [1.0] * len(dup_sub))
    a_masi_dup, _ = _weighted_krippendorff_masi_alpha(dup_masi, [1.0] * len(dup_masi))
    if abs(a_sub_int - a_sub_dup) >= 1e-12 or abs(a_masi_int - a_masi_dup) >= 1e-12:
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 3 (integer cluster-replication) failed")

    # Fixture 4: Zero-variance boundary (D_e = 0, D_o = 0 => 1.0; p_e = 1, p_o = 1 => 1.0).
    zv_sub, zv_flag1 = _weighted_krippendorff_nominal_alpha([["F1.1", "F1.1"]] * 4, [1.0] * 4)
    zv_masi, zv_flag2 = _weighted_krippendorff_masi_alpha([[frozenset({"c1"}), frozenset({"c1"})]] * 4, [1.0] * 4)
    zv_kap, zv_flag3 = _weighted_pair_stratified_cohen_kappa(["F"] * 4, ["F1.1"] * 4, ["F1.1"] * 4, [1.0] * 4)
    zv_fl, zv_flag4 = _weighted_fleiss_kappa_8rater([["F1.1"] * 8] * 4, [1.0] * 4)
    if (
        zv_sub != 1.0
        or zv_masi != 1.0
        or zv_kap != 1.0
        or zv_fl != 1.0
        or not (zv_flag1 and zv_flag2 and zv_flag3 and zv_flag4)
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 4 (zero-variance boundary) failed")
    return True


def _resolve_confined_method_study_file(root: Path, rel_str: str, label: str) -> Path:
    if not rel_str:
        raise ConceptGraphError(f"schema sync: missing {label}")
    rel_p = Path(rel_str)
    if rel_p.is_absolute() or ".." in rel_p.parts or len(rel_p.parts) < 3 or rel_p.parts[:2] != ("taxonomy", "method-study"):
        raise ConceptGraphError(f"schema sync: {label} {rel_str!r} must be a relative path confined under taxonomy/method-study/")
    cur = root
    if cur.is_symlink():
        raise ConceptGraphError(f"schema sync: symlinked root forbidden when resolving {label}")
    for part in rel_p.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ConceptGraphError(f"schema sync: symlink component {cur} is forbidden in {label}")
    ms_dir = (root / "taxonomy" / "method-study").resolve()
    resolved = cur.resolve()
    if not resolved.is_relative_to(ms_dir) or not resolved.is_file():
        raise ConceptGraphError(f"schema sync: {label} {rel_str!r} does not resolve to a regular file inside taxonomy/method-study/")
    return resolved


def _area_to_stage2_pair(area_code: str) -> str:
    if area_code in ("T1", "T2"):
        return area_code
    if area_code and area_code[0] in ("F", "S", "P", "Q"):
        return area_code[0]
    raise ConceptGraphError(f"schema sync: unknown area_code {area_code!r} in Stage 2 verification")


def _validate_person_record(
    rec: object,
    ctx: str,
    seen_pids: set[str],
    *,
    require_non_editor: bool = False,
) -> dict:
    if not isinstance(rec, dict):
        raise ConceptGraphError(f"schema sync: {ctx} must be a person mapping")
    pid = str(rec.get("person_id") or "").strip()
    inst = norm_person_or_org_token(str(rec.get("institution") or ""), is_org=True)
    if (
        len(pid) < 3
        or not inst
        or rec.get("five_year_coi_attested") is not True
        or not isinstance(rec.get("editor_institution"), bool)
        or not isinstance(rec.get("editor_coauthor_5yr"), bool)
    ):
        raise ConceptGraphError(
            f"schema sync: {ctx} requires person_id, institution, five_year_coi_attested=true, and boolean COI flags"
        )
    if pid in seen_pids:
        raise ConceptGraphError(f"schema sync: duplicate person_id {pid!r} across governance/indexing pools at {ctx}")
    if require_non_editor and (rec["editor_institution"] or rec["editor_coauthor_5yr"]):
        raise ConceptGraphError(f"schema sync: {ctx} ({pid!r}) must be independent of Editor institution and 5-year co-authorship")
    seen_pids.add(pid)
    return {**rec, "_norm_inst": inst, "_pid": pid}


def verify_release_governance_receipt(
    rel_rec: object,
    target_version: str,
    expected_tax_sha: str,
    expected_non_guides_sha: str | None = None,
    *,
    root: Path | None = None,
    candidate_tax_doc: dict | None = None,
    burned_doc: dict | None = None,
    prov_concepts: dict[str, dict] | None = None,
) -> None:
    """Enforce candidate-experts.md §2–§4 Editorial Board and 12-human Stage 2 gates on non-draft releases."""
    major, minor, _patch_v, pre_type, _rc_num = parse_taxonomy_semver(target_version)
    if pre_type == 0:
        return
    if major < 1:
        raise ConceptGraphError(
            f"schema sync: 0.x version {target_version!r} must carry '-draft' suffix; citable releases and release candidates start at 1.0-rc1 / 1.0"
        )
    if not isinstance(rel_rec, dict) or rel_rec.get("schema_version") != 1:
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} (major >= 1) requires schema_version=1 release_governance_receipt"
        )
    if str(rel_rec.get("target_version") or "").strip() != target_version:
        raise ConceptGraphError("schema sync: release_governance_receipt target_version disagrees with taxonomy_version")
    if str(rel_rec.get("candidate_taxonomy_sha256") or "").strip() != expected_tax_sha:
        raise ConceptGraphError("schema sync: release_governance_receipt candidate_taxonomy_sha256 disagrees with taxonomy_sha256")
    if expected_non_guides_sha is not None:
        if str(rel_rec.get("candidate_non_guides_taxonomy_sha256") or "").strip() != expected_non_guides_sha:
            raise ConceptGraphError(
                "schema sync: release_governance_receipt candidate_non_guides_taxonomy_sha256 disagrees with non_guides_taxonomy_sha256"
            )

    board = rel_rec.get("board_signoff")
    if (
        not isinstance(board, dict)
        or board.get("seated_voting_members") != 7
        or not isinstance(board.get("affirmative_votes"), int)
        or isinstance(board.get("affirmative_votes"), bool)
        or not (5 <= board["affirmative_votes"] <= 7)
        or board.get("v1_approved") is not True
        or board.get("v2_approved") is not True
        or board.get("sub_area_specialist_reports") != 13
        or not isinstance(board.get("active_editor_institution_voters"), int)
        or isinstance(board.get("active_editor_institution_voters"), bool)
        or not (0 <= board["active_editor_institution_voters"] <= 1)
        or not isinstance(board.get("active_editor_coauthor_voters"), int)
        or isinstance(board.get("active_editor_coauthor_voters"), bool)
        or not (0 <= board["active_editor_coauthor_voters"] <= 2)
        or not _HEX64_RE.match(str(board.get("charter_sha256") or ""))
        or not _HEX64_RE.match(str(board.get("vote_record_sha256") or ""))
    ):
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} requires board_signoff matching candidate-experts.md §2–§3 "
            "(7 seated voting seats, >=5 affirmative votes, v1_approved=true, v2_approved=true, 13 specialist reports, active caps <=1/<=2)"
        )
    hs = rel_rec.get("human_study_stage2")
    if (
        not isinstance(hs, dict)
        or not isinstance(hs.get("n_items"), int)
        or isinstance(hs.get("n_items"), bool)
        or hs["n_items"] != 240
        or hs.get("n_raters") != 12
        or not isinstance(hs.get("effective_lineage_components"), int)
        or isinstance(hs.get("effective_lineage_components"), bool)
        or hs["effective_lineage_components"] < 12
        or not isinstance(hs.get("bootstrap_resamples"), int)
        or isinstance(hs.get("bootstrap_resamples"), bool)
        or hs["bootstrap_resamples"] != 10000
        or not _is_finite_num(hs.get("alpha_subject"))
        or not (0.65 <= float(hs["alpha_subject"]) <= 1.0)
        or not _is_finite_num(hs.get("kappa_bar_subject"))
        or not (0.65 <= float(hs["kappa_bar_subject"]) <= 1.0)
        or not _is_finite_num(hs.get("cond_hooper"))
        or not (0.45 <= float(hs["cond_hooper"]) <= 1.0)
        or not _is_finite_num(hs.get("all_hooper"))
        or not (0.40 <= float(hs["all_hooper"]) <= 1.0)
        or not _is_finite_num(hs.get("cond_rolling"))
        or not (0.60 <= float(hs["cond_rolling"]) <= 1.0)
        or not _is_finite_num(hs.get("all_rolling"))
        or not (0.55 <= float(hs["all_rolling"]) <= 1.0)
        or not _is_finite_num(hs.get("alpha_masi"))
        or not (0.45 <= float(hs["alpha_masi"]) <= 1.0)
        or not _is_finite_num(hs.get("empty_rate"))
        or not (0.0 <= float(hs["empty_rate"]) <= 0.20)
        or not isinstance(hs.get("min_area_both_non_empty"), int)
        or isinstance(hs.get("min_area_both_non_empty"), bool)
        or not (8 <= hs["min_area_both_non_empty"] <= 12)
        or not _is_finite_num(hs.get("p_max"))
        or not (0.0 < float(hs["p_max"]) <= 0.15)
        or not isinstance(hs.get("recalibration_rerun_index"), int)
        or isinstance(hs.get("recalibration_rerun_index"), bool)
        or hs["recalibration_rerun_index"] not in (0, 1)
        or not _HEX64_RE.match(str(hs.get("study_bundle_sha256") or ""))
    ):
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} requires finite Stage 2 human study receipt matching candidate-experts.md §4.3 "
            "(n_items==240, n_raters==12, >=12 components, 10000 resamples, alpha_subject>=0.65, kappa_bar_subject>=0.65, "
            "cond_hooper>=0.45, all_hooper>=0.40, cond_rolling>=0.60, all_rolling>=0.55, alpha_masi>=0.45, empty_rate<=0.20, "
            "min_area_both_non_empty>=8, p_max<=0.15, recalibration_rerun_index in {0, 1})"
        )

    run_stage2_estimator_self_tests()

    eff_root = root if root is not None else REPO_ROOT_PATH
    gov_rel = str(rel_rec.get("governance_bundle_relpath") or "").strip()
    gov_sha_decl = str(rel_rec.get("governance_bundle_sha256") or "").strip()
    if not _HEX64_RE.match(gov_sha_decl) or gov_sha_decl != str(hs["study_bundle_sha256"]):
        raise ConceptGraphError(
            "schema sync: release_governance_receipt requires governance_bundle_sha256 matching human_study_stage2.study_bundle_sha256"
        )
    gov_path = _resolve_confined_method_study_file(eff_root, gov_rel, "governance_bundle_relpath")
    gov_bytes = gov_path.read_bytes()
    if hashlib.sha256(gov_bytes).hexdigest() != gov_sha_decl:
        raise ConceptGraphError("schema sync: governance_bundle_relpath SHA-256 disagrees with governance_bundle_sha256")
    gov_bundle = json.loads(gov_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
    if (
        not isinstance(gov_bundle, dict)
        or gov_bundle.get("schema_version") != 1
        or str(gov_bundle.get("target_version") or "").strip() != target_version
        or str(gov_bundle.get("candidate_taxonomy_sha256") or "").strip() != expected_tax_sha
    ):
        raise ConceptGraphError("schema sync: governance bundle header disagrees with target_version or candidate_taxonomy_sha256")
    if expected_non_guides_sha is not None and str(gov_bundle.get("candidate_non_guides_taxonomy_sha256") or "").strip() != expected_non_guides_sha:
        raise ConceptGraphError("schema sync: governance bundle candidate_non_guides_taxonomy_sha256 mismatch")

    eff_burned = burned_doc if burned_doc is not None else _load_burned_clusters(eff_root)
    if candidate_tax_doc is not None:
        eff_cand = candidate_tax_doc
    else:
        from .field_graph import load_taxonomy_yaml

        eff_cand = load_taxonomy_yaml(eff_root / "taxonomy/ml-systems-taxonomy.yml")

    # 1. Verify charter attestation, distinct >=46 person roster, COI caps, 7-seat votes, T1+T2 concurrence, & 20-area coverage.
    charter_att = gov_bundle.get("charter_attestation")
    vote_rec = gov_bundle.get("vote_record")
    if not isinstance(charter_att, dict) or not charter_att or not isinstance(vote_rec, dict) or not vote_rec:
        raise ConceptGraphError("schema sync: governance bundle missing charter_attestation or vote_record")
    if hashlib.sha256(canonical_taxonomy_json(charter_att).encode("utf-8")).hexdigest() != board["charter_sha256"]:
        raise ConceptGraphError("schema sync: charter_attestation SHA-256 disagrees with board_signoff.charter_sha256")
    if hashlib.sha256(canonical_taxonomy_json(vote_rec).encode("utf-8")).hexdigest() != board["vote_record_sha256"]:
        raise ConceptGraphError("schema sync: vote_record SHA-256 disagrees with board_signoff.vote_record_sha256")

    seen_pids: set[str] = set()
    conv = _validate_person_record(gov_bundle.get("convenor"), "convenor", seen_pids)
    if conv.get("voting") is not False:
        raise ConceptGraphError("schema sync: convenor must be non-voting (voting=false)")

    nom_panel = gov_bundle.get("nominating_panel")
    if not isinstance(nom_panel, list) or len(nom_panel) != 3:
        raise ConceptGraphError("schema sync: nominating_panel must list 3 independent members")
    for idx_np, np_rec in enumerate(nom_panel):
        _validate_person_record(np_rec, f"nominating_panel[{idx_np}]", seen_pids, require_non_editor=True)

    steer = _validate_person_record(gov_bundle.get("steering_advisor"), "steering_advisor", seen_pids)
    if steer.get("voting") is not False:
        raise ConceptGraphError("schema sync: steering_advisor must be non-voting (voting=false)")

    standing_raw = gov_bundle.get("standing_roster")
    if not isinstance(standing_raw, list) or len(standing_raw) != 16:
        raise ConceptGraphError("schema sync: standing_roster must list 16 standing board members/alternates")
    standing_by_pid: dict[str, dict] = {}
    for idx_sr, sr_rec in enumerate(standing_raw):
        vp = _validate_person_record(sr_rec, f"standing_roster[{idx_sr}]", seen_pids)
        seat = str(vp.get("seat") or "").strip()
        role = str(vp.get("role") or "").strip()
        if seat not in _BOARD_SEATS_EXPECTED or role not in ("chair", "co_chair", "alternate"):
            raise ConceptGraphError(f"schema sync: standing_roster[{idx_sr}] invalid seat={seat!r} or role={role!r}")
        standing_by_pid[vp["_pid"]] = vp
    if sum(1 for m in standing_by_pid.values() if m["editor_institution"]) > 2:
        raise ConceptGraphError("schema sync: standing_roster exceeds <=2 Editor-institution cap")
    if sum(1 for m in standing_by_pid.values() if m["editor_coauthor_5yr"]) > 3:
        raise ConceptGraphError("schema sync: standing_roster exceeds <=3 5-year Editor-coauthor cap")

    specs_raw = gov_bundle.get("sub_area_specialists")
    if not isinstance(specs_raw, list) or len(specs_raw) != 13:
        raise ConceptGraphError("schema sync: sub_area_specialists must list all 13 mandatory sub-area specialists")
    seen_spec_slots: set[str] = set()
    specialist_covered_areas: set[str] = set()
    for idx_sp, sp_rec in enumerate(specs_raw):
        vp = _validate_person_record(sp_rec, f"sub_area_specialists[{idx_sp}]", seen_pids)
        slot = str(vp.get("specialist_slot") or "").strip()
        if slot not in _SPECIALIST_SLOTS_EXPECTED or slot in seen_spec_slots:
            raise ConceptGraphError(f"schema sync: invalid or duplicate specialist_slot {slot!r}")
        if vp.get("report_submitted") is not True or not _HEX64_RE.match(str(vp.get("report_sha256") or "")):
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} missing submitted report_sha256")
        seen_spec_slots.add(slot)
        specialist_covered_areas.add("Q4" if slot in ("Q4a", "Q4b") else slot)

    chair_signoffs = gov_bundle.get("chair_area_signoffs")
    if not isinstance(chair_signoffs, dict) or set(chair_signoffs.keys()) != set(_CHAIR_SIGN_OFF_AREAS_TO_SEAT.keys()):
        raise ConceptGraphError(
            "schema sync: chair_area_signoffs must cover exact 8 non-specialist areas {F1,F2,S1,S2,S3,P4,T1,T2}"
        )
    for acode, req_seat in _CHAIR_SIGN_OFF_AREAS_TO_SEAT.items():
        cs_entry = chair_signoffs[acode]
        cs_pid = str(cs_entry.get("signer_person_id") if isinstance(cs_entry, dict) else cs_entry or "").strip()
        if cs_pid not in standing_by_pid or standing_by_pid[cs_pid].get("seat") != req_seat:
            raise ConceptGraphError(f"schema sync: chair_area_signoffs[{acode!r}] must be signed by a standing member of seat {req_seat}")
    expected_20_areas = {str(a["code"]) for a in eff_cand.get("areas", [])}
    if (specialist_covered_areas | set(chair_signoffs.keys())) != expected_20_areas:
        raise ConceptGraphError("schema sync: 13 sub_area_specialists + 8 chair_area_signoffs do not cover all 20 taxonomy areas")

    indexers_raw = gov_bundle.get("human_indexers")
    if not isinstance(indexers_raw, list) or len(indexers_raw) != 12:
        raise ConceptGraphError("schema sync: human_indexers must list 12 distinct human indexers across 6 pairs")
    indexers_by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    indexer_inst_counts: collections.Counter[str] = collections.Counter()
    for idx_hi, hi_rec in enumerate(indexers_raw):
        vp = _validate_person_record(hi_rec, f"human_indexers[{idx_hi}]", seen_pids, require_non_editor=True)
        pair_id = str(vp.get("pair_id") or "").strip()
        if pair_id not in _STAGE2_PAIR_ORDER or vp.get("pair_coauthored_5yr") is not False:
            raise ConceptGraphError(f"schema sync: human_indexers[{idx_hi}] requires valid pair_id and pair_coauthored_5yr=false")
        indexers_by_pair[pair_id].append(vp)
        indexer_inst_counts[vp["_norm_inst"]] += 1
    if any(cnt > 2 for cnt in indexer_inst_counts.values()):
        raise ConceptGraphError("schema sync: human_indexers exceeds <=2 indexers per institution cap")
    for p_id in _STAGE2_PAIR_ORDER:
        pair_members = indexers_by_pair.get(p_id, [])
        if len(pair_members) != 2 or pair_members[0]["_norm_inst"] == pair_members[1]["_norm_inst"]:
            raise ConceptGraphError(f"schema sync: pair {p_id} must have 2 indexers from distinct institutions")

    adhoc_raw = gov_bundle.get("adhoc_voters") or []
    if not isinstance(adhoc_raw, list):
        raise ConceptGraphError("schema sync: adhoc_voters must be a list")
    adhoc_by_pid: dict[str, dict] = {}
    for idx_av, av_rec in enumerate(adhoc_raw):
        vp = _validate_person_record(av_rec, f"adhoc_voters[{idx_av}]", seen_pids, require_non_editor=True)
        if str(vp.get("seat") or "").strip() not in _BOARD_SEATS_EXPECTED:
            raise ConceptGraphError(f"schema sync: adhoc_voters[{idx_av}] missing valid seat")
        adhoc_by_pid[vp["_pid"]] = vp

    if len(seen_pids) < 46:
        raise ConceptGraphError("schema sync: governance bundle must have >=46 distinct natural persons across all pools")

    seat_votes = vote_rec.get("seat_votes")
    if not isinstance(seat_votes, dict) or set(seat_votes.keys()) != _BOARD_SEATS_EXPECTED:
        raise ConceptGraphError("schema sync: vote_record.seat_votes must record all 7 voting seats")
    active_voters: list[dict] = []
    seen_active_voters: set[str] = set()
    aff_count = 0
    for seat_k, sv in sorted(seat_votes.items()):
        if not isinstance(sv, dict) or sv.get("recused") is not False or sv.get("vote") not in ("approve", "reject", "abstain"):
            raise ConceptGraphError(f"schema sync: vote_record.seat_votes[{seat_k!r}] invalid vote or recusal flag")
        vpids_here = [str(sv.get("voter_person_id") or "").strip()]
        if seat_k == "T" and sv.get("t2_voter_person_id"):
            vpids_here.append(str(sv.get("t2_voter_person_id") or "").strip())
        for vpid in vpids_here:
            if not vpid or vpid in seen_active_voters:
                raise ConceptGraphError(f"schema sync: person {vpid!r} cannot cast votes for multiple seats")
            seen_active_voters.add(vpid)
            vperson = standing_by_pid.get(vpid) or adhoc_by_pid.get(vpid)
            if vperson is None or str(vperson.get("seat")) != seat_k:
                raise ConceptGraphError(f"schema sync: voter {vpid!r} is not an eligible member/replacement for seat {seat_k}")
            active_voters.append(vperson)
        if sv["vote"] == "approve":
            aff_count += 1
    act_inst = sum(1 for v in active_voters if v["editor_institution"])
    act_coauth = sum(1 for v in active_voters if v["editor_coauthor_5yr"])
    if (
        aff_count != board["affirmative_votes"]
        or aff_count < 5
        or seat_votes["V1"]["vote"] != "approve"
        or seat_votes["V2"]["vote"] != "approve"
        or act_inst != board["active_editor_institution_voters"]
        or act_inst > 1
        or act_coauth != board["active_editor_coauthor_voters"]
        or act_coauth > 2
        or vote_rec.get("unresolved_binding_rejections") != []
    ):
        raise ConceptGraphError("schema sync: vote_record fails supermajority, V1/V2 concurrence, COI cap, or unresolved rejection check")
    if seat_votes["T"]["vote"] == "approve":
        if vote_rec.get("seat_t_t1_concurred") is not True or vote_rec.get("seat_t_t2_concurred") is not True:
            raise ConceptGraphError("schema sync: affirmative Seat T vote requires both seat_t_t1_concurred=true and seat_t_t2_concurred=true")

    # 2. Verify pre-1.0 legacy alias dispositions across all 7 pinned legacy concepts and all unattested active legacy aliases.
    leg_disp = gov_bundle.get("v1_legacy_alias_dispositions")
    if not isinstance(leg_disp, dict) or not set(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS.keys()).issubset(leg_disp.keys()):
        raise ConceptGraphError("schema sync: governance bundle must record v1_legacy_alias_dispositions covering all 7 pinned legacy concepts")
    cand_concepts_by_id = {c["id"]: c for c in eff_cand.get("concepts", [])}
    for gcid, req_aliases in PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS.items():
        c_disp = leg_disp.get(gcid)
        if not isinstance(c_disp, dict) or not set(req_aliases).issubset(c_disp.keys()):
            raise ConceptGraphError(f"schema sync: v1_legacy_alias_dispositions[{gcid!r}] must cover {list(req_aliases)}")
        active_norms_g = {_norm_prov_tokens(str(a)) for a in ((cand_concepts_by_id.get(gcid) or {}).get("aliases") or [])}
        for galias in req_aliases:
            drec = c_disp[galias]
            if (
                not isinstance(drec, dict)
                or drec.get("disposition") not in ("uphold", "remove", "reclassify-or-split")
                or drec.get("v1_signed_off") is not True
                or len(str(drec.get("rationale") or "").strip()) < 15
            ):
                raise ConceptGraphError(f"schema sync: invalid v1_legacy_alias_dispositions entry on {gcid} alias {galias!r}")
            is_still_active = _norm_prov_tokens(galias) in active_norms_g
            if is_still_active and drec["disposition"] != "uphold":
                raise ConceptGraphError(f"schema sync: active legacy alias {galias!r} on {gcid} requires disposition='uphold'")
            if drec["disposition"] == "remove" and is_still_active:
                raise ConceptGraphError(f"schema sync: legacy alias {galias!r} on {gcid} marked 'remove' is still active")

    for cid_cand, c_node in cand_concepts_by_id.items():
        att_meta = ((prov_concepts.get(cid_cand) if isinstance(prov_concepts, dict) else None) or {}).get("alias_metadata") or {}
        att_norms = {_norm_prov_tokens(str(ak)) for ak in att_meta} if isinstance(att_meta, dict) else set()
        for active_al in c_node.get("aliases") or []:
            al_str = str(active_al)
            if _norm_prov_tokens(al_str) in att_norms:
                continue
            c_disp_all = leg_disp.get(cid_cand)
            drec_all = c_disp_all.get(al_str) if isinstance(c_disp_all, dict) else None
            if (
                not isinstance(drec_all, dict)
                or drec_all.get("disposition") != "uphold"
                or drec_all.get("v1_signed_off") is not True
                or len(str(drec_all.get("rationale") or "").strip()) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: unattested legacy alias {al_str!r} on {cid_cand} requires v1_legacy_alias_dispositions[{cid_cand!r}][{al_str!r}] with disposition='uphold', v1_signed_off=true, and >=15-char rationale"
                )

    # 3. Verify calibrated seeded-gap sensitivity >= 85% across >= 10 ablated concepts.
    sg_audit = gov_bundle.get("seeded_gap_audit")
    sg_items = sg_audit.get("seeded_gap_items") if isinstance(sg_audit, dict) else None
    if not isinstance(sg_items, list) or len(sg_items) < 10:
        raise ConceptGraphError("schema sync: non-draft release requires seeded_gap_audit.seeded_gap_items with >=10 probes")
    ablated_cids: set[str] = set()
    degraded_cnt = 0
    for idx_sg, sgi in enumerate(sg_items):
        if (
            not isinstance(sgi, dict)
            or not str(sgi.get("unit_id") or "").strip()
            or not str(sgi.get("judge_id") or "").strip()
            or not str(sgi.get("ablated_concept_id") or "").strip()
            or not _is_finite_num(sgi.get("baseline_score"))
            or not _is_finite_num(sgi.get("ablated_score"))
        ):
            raise ConceptGraphError(f"schema sync: malformed seeded_gap_items[{idx_sg}]")
        bs = float(sgi["baseline_score"])
        as_v = float(sgi["ablated_score"])
        if bs not in (0.5, 1.0) or as_v not in (0.0, 0.5, 1.0) or as_v > bs:
            raise ConceptGraphError(f"schema sync: invalid ordinal scores in seeded_gap_items[{idx_sg}]")
        ablated_cids.add(str(sgi["ablated_concept_id"]).strip())
        if as_v < bs:
            degraded_cnt += 1
    if len(ablated_cids) < 10:
        raise ConceptGraphError("schema sync: seeded_gap_audit must span >=10 distinct ablated_concept_ids")
    sg_sens = degraded_cnt / len(sg_items)
    if sg_sens < 0.85:
        raise ConceptGraphError(f"schema sync: non-draft release seeded-gap sensitivity {sg_sens:.4f} < 0.85 floor")
    if "seeded_gap_sensitivity" in hs:
        if not _is_finite_num(hs["seeded_gap_sensitivity"]) or abs(float(hs["seeded_gap_sensitivity"]) - sg_sens) > 1e-9:
            raise ConceptGraphError("schema sync: human_study_stage2.seeded_gap_sensitivity disagrees with recomputed seeded_gap_items")

    # 4. Verify Stage 2 human study raw annotations, rerun history, lineage disjointness, routing, & dual bootstrap.
    s2 = gov_bundle.get("stage2_study")
    if not isinstance(s2, dict):
        raise ConceptGraphError("schema sync: governance bundle missing stage2_study")
    s1_items = s2.get("stage1_calibration_items")
    s2_items = s2.get("stage2_items")
    recal_items = s2.get("recalibration_slice_items") or []
    if not isinstance(s1_items, list) or len(s1_items) != 30 or not isinstance(s2_items, list) or len(s2_items) != 240 or not isinstance(recal_items, list):
        raise ConceptGraphError("schema sync: stage2_study requires 30 stage1_calibration_items and 240 stage2_items")

    rerun_idx = hs["recalibration_rerun_index"]
    cycle_mm = f"{major}.{minor}"
    ledger_s2_attempts = [
        a for a in (eff_burned.get("stage2_human_attempts") or [])
        if isinstance(a, dict) and str(a.get("target_major_minor") or "") == cycle_mm
    ]
    if len(ledger_s2_attempts) > 2:
        raise ConceptGraphError(f"schema sync: burned-clusters stage2_human_attempts exceeds max 1 rerun for {cycle_mm}")
    prior_attempt_units: list[dict] = []
    if rerun_idx == 0:
        for att in ledger_s2_attempts:
            if att.get("recalibration_rerun_index") == 1 or (
                att.get("recalibration_rerun_index") == 0 and att.get("study_bundle_sha256") != gov_sha_decl
            ):
                raise ConceptGraphError(
                    "schema sync: recalibration_rerun_index=0 conflicts with existing stage2_human_attempts in burned-clusters.json"
                )
    else:
        prior_rel = str(s2.get("prior_failed_bundle_relpath") or "").strip()
        prior_sha = str(s2.get("prior_failed_bundle_sha256") or "").strip()
        if not _HEX64_RE.match(prior_sha) or prior_sha == gov_sha_decl:
            raise ConceptGraphError("schema sync: recalibration_rerun_index=1 requires valid distinct prior_failed_bundle_sha256")
        if ledger_s2_attempts:
            att0 = [a for a in ledger_s2_attempts if a.get("recalibration_rerun_index") == 0]
            if not att0 or att0[0].get("study_bundle_sha256") != prior_sha:
                raise ConceptGraphError("schema sync: prior_failed_bundle_sha256 disagrees with burned-clusters stage2_human_attempts[0]")
        prior_path = _resolve_confined_method_study_file(eff_root, prior_rel, "prior_failed_bundle_relpath")
        prior_bytes = prior_path.read_bytes()
        if hashlib.sha256(prior_bytes).hexdigest() != prior_sha:
            raise ConceptGraphError("schema sync: prior_failed_bundle_relpath SHA-256 mismatch")
        prior_bundle = json.loads(prior_bytes.decode("utf-8"), parse_constant=_reject_json_constant)
        prior_s2 = (prior_bundle.get("stage2_study") or {}) if isinstance(prior_bundle, dict) else {}
        if prior_s2.get("recalibration_rerun_index", 0) != 0:
            raise ConceptGraphError("schema sync: prior_failed_bundle must have recalibration_rerun_index=0")
        for lst_k in ("stage1_calibration_items", "stage2_items", "recalibration_slice_items"):
            for u_p in (prior_s2.get(lst_k) or []):
                if isinstance(u_p, dict):
                    prior_attempt_units.append(u_p)
        if not prior_attempt_units or not recal_items:
            raise ConceptGraphError(
                "schema sync: recalibration_rerun_index=1 requires non-empty prior_failed_bundle items and recalibration_slice_items"
            )

    all_study_units = [*s1_items, *s2_items, *recal_items, *prior_attempt_units]
    closure = compute_lineage_components(all_study_units, eff_burned, include_all_exposed_meta=True)
    merges: dict[str, str] = closure["merges"]  # type: ignore[assignment]
    pf_to_comp: dict[str, str] = closure["pf_to_component"]  # type: ignore[assignment]
    comp_ed: set[str] = closure["component_editor_affiliated"]  # type: ignore[assignment]
    internal_set: set[str] = closure["internal_clusters"]  # type: ignore[assignment]

    pre_burned_comps: set[str] = set()
    exp_meta_map = eff_burned.get("exposed_cluster_metadata") or {}
    for raw_b_str in eff_burned.get("burned_or_exposed_cluster_ids") or []:
        s_bpf = str(raw_b_str or "").strip()
        if not s_bpf:
            continue
        canon_bpf = merges.get(s_bpf, s_bpf)
        meta_bpf = exp_meta_map.get(canon_bpf) if isinstance(exp_meta_map, dict) else None
        if isinstance(meta_bpf, dict) and str(meta_bpf.get("burned_by_stage2_bundle_sha256") or "") == gov_sha_decl:
            continue
        if canon_bpf in pf_to_comp:
            pre_burned_comps.add(pf_to_comp[canon_bpf])
    for bc in eff_burned.get("burned_clusters") or []:
        if not isinstance(bc, dict):
            continue
        if str(bc.get("burned_by_stage2_bundle_sha256") or "") == gov_sha_decl:
            continue
        raw_bpf = str(bc.get("cluster_id") or "").strip()
        if raw_bpf:
            canon_bpf = merges.get(raw_bpf, raw_bpf)
            if canon_bpf in pf_to_comp:
                pre_burned_comps.add(pf_to_comp[canon_bpf])

    def _unit_ext_comp(u_obj: dict, label_ctx: str) -> str:
        raw_pf = str(u_obj.get("provenance_family_id") or "").strip()
        if not raw_pf or u_obj.get("origin") != "external" or bool(u_obj.get("editor_affiliated", False)):
            raise ConceptGraphError(f"schema sync: {label_ctx} must be non-editor-affiliated origin='external'")
        if not (u_obj.get("authors") or u_obj.get("institutional_organizations") or u_obj.get("institutional_organization")):
            raise ConceptGraphError(f"schema sync: {label_ctx} missing authors/institutional_organizations metadata")
        canon_pf = merges.get(raw_pf, raw_pf)
        if canon_pf in internal_set:
            raise ConceptGraphError(f"schema sync: {label_ctx} uses internal cluster {canon_pf!r}")
        comp_id = pf_to_comp[canon_pf]
        if comp_id == "eff_lineage:internal" or comp_id in comp_ed:
            raise ConceptGraphError(f"schema sync: {label_ctx} resolves to internal or editor-affiliated lineage {comp_id!r}")
        if comp_id in pre_burned_comps:
            raise ConceptGraphError(f"schema sync: {label_ctx} uses burned lineage component {comp_id!r}")
        return comp_id

    s1_comps: set[str] = set()
    s1_pair_counts: collections.Counter[str] = collections.Counter()
    seen_unit_ids: set[str] = set()
    for idx_u, u1 in enumerate(s1_items):
        if not isinstance(u1, dict):
            raise ConceptGraphError(f"schema sync: malformed stage1_calibration_items[{idx_u}]")
        uid1 = str(u1.get("unit_id") or "").strip()
        p1 = str(u1.get("pair_id") or "").strip()
        if not uid1 or uid1 in seen_unit_ids or p1 not in _STAGE2_PAIR_ORDER:
            raise ConceptGraphError(f"schema sync: invalid or duplicate unit_id/pair_id in stage1_calibration_items[{idx_u}]")
        seen_unit_ids.add(uid1)
        s1_pair_counts[p1] += 1
        s1_comps.add(_unit_ext_comp(u1, f"stage1_calibration_items[{idx_u}]"))
    if any(s1_pair_counts[p] != 5 for p in _STAGE2_PAIR_ORDER):
        raise ConceptGraphError("schema sync: stage1_calibration_items must allocate 5 items to each of the 6 pairs")

    recal_comps = {_unit_ext_comp(ur, f"recalibration_slice_items[{i}]") for i, ur in enumerate(recal_items)}
    prior_comps: set[str] = set()
    for ip, up in enumerate(prior_attempt_units):
        raw_ppf = str(up.get("provenance_family_id") or "").strip()
        if raw_ppf:
            cppf = merges.get(raw_ppf, raw_ppf)
            if cppf in pf_to_comp:
                prior_comps.add(pf_to_comp[cppf])

    valid_subjects = {str(r["id"]) for a in eff_cand.get("areas", []) for r in a.get("rows", [])}
    valid_concepts = {str(c["id"]) for c in eff_cand.get("concepts", [])}
    non_leaf_concepts = {
        str(e["target"])
        for e in eff_cand.get("relationships", [])
        if e.get("relation") == "specializes" and str(e.get("source")) in valid_concepts
    }
    pair_rater_sets = {p: {m["_pid"] for m in indexers_by_pair[p]} for p in _STAGE2_PAIR_ORDER}

    items_by_area: dict[str, list[dict]] = collections.defaultdict(list)
    for idx_2, u2 in enumerate(s2_items):
        if not isinstance(u2, dict):
            raise ConceptGraphError(f"schema sync: malformed stage2_items[{idx_2}]")
        uid2 = str(u2.get("unit_id") or "").strip()
        acode = str(u2.get("area_code") or "").strip()
        if not uid2 or uid2 in seen_unit_ids or acode not in expected_20_areas:
            raise ConceptGraphError(f"schema sync: invalid or duplicate unit_id/area_code in stage2_items[{idx_2}]")
        seen_unit_ids.add(uid2)
        comp2 = _unit_ext_comp(u2, f"stage2_items[{idx_2}]")
        if comp2 in s1_comps or comp2 in recal_comps or comp2 in prior_comps:
            raise ConceptGraphError(
                f"schema sync: stage2_items[{idx_2}] ({uid2}) shares effective_lineage_component_id {comp2} with Stage 1, recalibration, or prior attempt"
            )
        items_by_area[acode].append({**u2, "_comp": comp2})

    if set(items_by_area.keys()) != expected_20_areas:
        raise ConceptGraphError("schema sync: stage2_items must cover all 20 taxonomy areas")

    ordered_s2: list[dict] = []
    area_item_indices: list[list[int]] = []
    anchors_by_group: dict[str, list[dict]] = collections.defaultdict(list)
    min_area_both_non_empty = 12

    def _check_pair_eval(pe_obj: object, expected_pair: str, ctx_lbl: str) -> tuple[str, str, frozenset[str], frozenset[str]]:
        if not isinstance(pe_obj, dict) or str(pe_obj.get("pair_id") or "").strip() != expected_pair:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} pair_id must equal {expected_pair!r}")
        r1 = str(pe_obj.get("rater_1") or "").strip()
        r2 = str(pe_obj.get("rater_2") or "").strip()
        if {r1, r2} != pair_rater_sets[expected_pair]:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} raters {(r1, r2)} disagree with human_indexers[{expected_pair}]")
        sb1 = str(pe_obj.get("subject_1") or "").strip()
        sb2 = str(pe_obj.get("subject_2") or "").strip()
        if sb1 not in valid_subjects or sb2 not in valid_subjects:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} invalid subject ID")
        raw_c1 = pe_obj.get("concepts_1")
        raw_c2 = pe_obj.get("concepts_2")
        if (
            not isinstance(raw_c1, list)
            or not isinstance(raw_c2, list)
            or len(raw_c1) > 3
            or len(raw_c2) > 3
            or len(set(raw_c1)) != len(raw_c1)
            or len(set(raw_c2)) != len(raw_c2)
            or not set(raw_c1).issubset(valid_concepts)
            or not set(raw_c2).issubset(valid_concepts)
        ):
            raise ConceptGraphError(f"schema sync: {ctx_lbl} concepts_1/concepts_2 must be 0..3 distinct active concept IDs")
        return (sb1, sb2, frozenset(str(x) for x in raw_c1), frozenset(str(x) for x in raw_c2))

    for acode in sorted(expected_20_areas):
        alist = sorted(items_by_area[acode], key=lambda x: (str(x["area_code"]), str(x["unit_id"])))
        if len(alist) != 12:
            raise ConceptGraphError(f"schema sync: area {acode} has {len(alist)} Stage 2 items (expected 12)")
        if len({x["_comp"] for x in alist}) != 12:
            raise ConceptGraphError(f"schema sync: area {acode} fails <=1 item per effective_lineage_component_id cap")
        exp_pair = _area_to_stage2_pair(acode)
        idxs_here: list[int] = []
        area_non_empty = 0
        for pos_in_area, item in enumerate(alist):
            exp_anchor = pos_in_area < 2
            if bool(item.get("is_anchor", False)) != exp_anchor:
                raise ConceptGraphError(
                    f"schema sync: item {item['unit_id']} in {acode} is_anchor={item.get('is_anchor')} disagrees with deterministic first-2-per-area rule"
                )
            sb1, sb2, fs1, fs2 = _check_pair_eval(item.get("primary_evaluation"), exp_pair, f"stage2[{item['unit_id']}].primary")
            if fs1 and fs2:
                area_non_empty += 1
            rec_clean = {
                **item,
                "_group": exp_pair,
                "_sb1": sb1,
                "_sb2": sb2,
                "_fs1": fs1,
                "_fs2": fs2,
            }
            if exp_anchor:
                anchors_by_group[exp_pair].append(rec_clean)
            else:
                if item.get("secondary_evaluations"):
                    raise ConceptGraphError(f"schema sync: non-anchor item {item['unit_id']} cannot carry secondary_evaluations")
            idxs_here.append(len(ordered_s2))
            ordered_s2.append(rec_clean)
        area_item_indices.append(idxs_here)
        if area_non_empty < min_area_both_non_empty:
            min_area_both_non_empty = area_non_empty

    exp_sched = expected_stage2_anchor_schedule()
    ordered_anchors: list[dict] = []
    anchor_8subjs: list[list[str]] = []
    anchor_8sets: list[list[frozenset[str]]] = []
    for g_id in _STAGE2_PAIR_ORDER:
        g_ancs = sorted(anchors_by_group[g_id], key=lambda x: (str(x["area_code"]), str(x["unit_id"])))
        if len(g_ancs) != len(exp_sched[g_id]):
            raise ConceptGraphError(f"schema sync: group {g_id} anchor count mismatch")
        for k_pos, anc_item in enumerate(g_ancs):
            sec_list = anc_item.get("secondary_evaluations")
            exp_receivers = exp_sched[g_id][k_pos]
            if not isinstance(sec_list, list) or len(sec_list) != 3:
                raise ConceptGraphError(f"schema sync: anchor {anc_item['unit_id']} requires 3 secondary_evaluations")
            sec_by_p = {str(se.get("pair_id") or "").strip(): se for se in sec_list if isinstance(se, dict)}
            if tuple(sorted(sec_by_p.keys(), key=lambda p: _STAGE2_PAIR_ORDER.index(p) if p in _STAGE2_PAIR_ORDER else 99)) != exp_receivers:
                raise ConceptGraphError(
                    f"schema sync: anchor {anc_item['unit_id']} secondary receiver pairs {list(sec_by_p.keys())} disagree with schedule {list(exp_receivers)}"
                )
            subjs_8 = [anc_item["_sb1"], anc_item["_sb2"]]
            sets_8 = [anc_item["_fs1"], anc_item["_fs2"]]
            for rec_p in exp_receivers:
                s_b1, s_b2, f_s1, f_s2 = _check_pair_eval(sec_by_p[rec_p], rec_p, f"anchor[{anc_item['unit_id']}].sec[{rec_p}]")
                subjs_8.extend([s_b1, s_b2])
                sets_8.extend([f_s1, f_s2])
            ordered_anchors.append(anc_item)
            anchor_8subjs.append(subjs_8)
            anchor_8sets.append(sets_8)

    # 5. Recompute primary 240-item point estimates, 40-anchor 8-rater point estimates, and 10,000-resample bootstraps.
    w_240 = [1.0] * 240
    groups_240 = [x["_group"] for x in ordered_s2]
    s1_240 = [x["_sb1"] for x in ordered_s2]
    s2_240 = [x["_sb2"] for x in ordered_s2]
    fs1_240 = [x["_fs1"] for x in ordered_s2]
    fs2_240 = [x["_fs2"] for x in ordered_s2]
    both_non_empty_mask = [bool(a and b) for a, b in zip(fs1_240, fs2_240)]

    pt_alpha_subj, _ = _weighted_krippendorff_nominal_alpha([[a, b] for a, b in zip(s1_240, s2_240)], w_240)
    pt_kappa_bar, _ = _weighted_pair_stratified_cohen_kappa(groups_240, s1_240, s2_240, w_240)
    pt_hr = _weighted_hooper_rolling(fs1_240, fs2_240, w_240)
    pt_alpha_masi, _ = _weighted_krippendorff_masi_alpha([[a, b] for a, b in zip(fs1_240, fs2_240)], w_240)

    tag_counts: collections.Counter[str] = collections.Counter()
    total_tags = 0
    leaf_tags = 0
    for fs_pair in (fs1_240, fs2_240):
        for fs in fs_pair:
            for cid_t in fs:
                tag_counts[cid_t] += 1
                total_tags += 1
                if cid_t not in non_leaf_concepts:
                    leaf_tags += 1
    if total_tags == 0:
        raise ConceptGraphError("schema sync: Stage 2 total concept tag assignments cannot be 0")
    pt_p_max = max(tag_counts.values()) / total_tags
    pt_leaf_spec = leaf_tags / total_tags

    w_40 = [1.0] * 40
    pt_anc_alpha_subj, _ = _weighted_krippendorff_nominal_alpha(anchor_8subjs, w_40)
    pt_anc_alpha_masi, _ = _weighted_krippendorff_masi_alpha(anchor_8sets, w_40)
    pt_anc_fleiss, _ = _weighted_fleiss_kappa_8rater(anchor_8subjs, w_40)

    comp_list_240 = sorted({x["_comp"] for x in ordered_s2})
    m_eff = len(comp_list_240)
    if m_eff < 12 or hs["effective_lineage_components"] != m_eff:
        raise ConceptGraphError(
            f"schema sync: effective_lineage_components={hs['effective_lineage_components']} disagrees with recomputed M_eff={m_eff}"
        )

    # Precompute MASI pattern tables once for the 10,000 primary resamples.
    pat_list: list[frozenset[str]] = []
    pat_map: dict[frozenset[str], int] = {}
    enc_240: list[list[int]] = []
    dis_240: list[float] = []
    for a, b in zip(fs1_240, fs2_240):
        for fs in (a, b):
            if fs not in pat_map:
                pat_map[fs] = len(pat_list)
                pat_list.append(fs)
        enc_240.append([pat_map[a], pat_map[b]])
        dis_240.append(2.0 * (1.0 - _masi_similarity(a, b)))
    nz_pairs_240: list[tuple[int, int, float]] = []
    for p in range(len(pat_list)):
        for q in range(p + 1, len(pat_list)):
            sim_pq = _masi_similarity(pat_list[p], pat_list[q])
            if sim_pq > 0.0:
                nz_pairs_240.append((p, q, sim_pq))

    comp_to_idx_240 = {c: idx for idx, c in enumerate(comp_list_240)}
    item_comp_idx_240 = [comp_to_idx_240[x["_comp"]] for x in ordered_s2]
    rng_primary = random.Random(20270101)
    degen_primary = 0
    valid_primary = 0
    while valid_primary < 10000:
        mults = [0] * m_eff
        for _ in range(m_eff):
            mults[rng_primary.randrange(m_eff)] += 1
        if sum(1 for c in mults if c > 0) < 2:
            degen_primary += 1
            if degen_primary > 100:
                raise ConceptGraphError("schema sync: Stage 2 primary bootstrap exceeded degenerate_resample_count > 100")
            continue
        w_draw = [0.0] * 240
        area_zero = False
        for idxs_a in area_item_indices:
            raw_sum_a = sum(mults[item_comp_idx_240[i]] for i in idxs_a)
            if raw_sum_a == 0:
                area_zero = True
                break
            scale_a = 12.0 / raw_sum_a
            for i in idxs_a:
                m_i = mults[item_comp_idx_240[i]]
                if m_i:
                    w_draw[i] = m_i * scale_a
        if area_zero or sum(w_draw[i] for i in range(240) if both_non_empty_mask[i]) <= 0.0:
            degen_primary += 1
            if degen_primary > 100:
                raise ConceptGraphError("schema sync: Stage 2 primary bootstrap exceeded degenerate_resample_count > 100")
            continue
        bs_masi, _ = _fast_masi_alpha_from_precomputed(enc_240, dis_240, nz_pairs_240, len(pat_list), w_draw)
        if not math.isfinite(bs_masi) or bs_masi < -1.0 or bs_masi > 1.0:
            raise ConceptGraphError("schema sync: non-finite or out-of-bounds bootstrap alpha_masi")
        valid_primary += 1

    # Separate pooled 40-anchor cluster bootstrap (seed = 20270102).
    anc_comp_list = sorted({x["_comp"] for x in ordered_anchors})
    m_anc = len(anc_comp_list)
    if m_anc < 2:
        raise ConceptGraphError("schema sync: 40 anchors require >=2 distinct effective_lineage_component_ids")
    anc_comp_to_idx = {c: idx for idx, c in enumerate(anc_comp_list)}
    anc_comp_idx = [anc_comp_to_idx[x["_comp"]] for x in ordered_anchors]
    rng_anc = random.Random(20270102)
    degen_anc = 0
    valid_anc = 0
    while valid_anc < 10000:
        mults_a = [0] * m_anc
        for _ in range(m_anc):
            mults_a[rng_anc.randrange(m_anc)] += 1
        raw_anc = [mults_a[anc_comp_idx[i]] for i in range(40)]
        raw_anc_sum = sum(raw_anc)
        if sum(1 for c in mults_a if c > 0) < 2 or raw_anc_sum == 0:
            degen_anc += 1
            if degen_anc > 100:
                raise ConceptGraphError("schema sync: Stage 2 anchor bootstrap exceeded anchor_degenerate_resample_count > 100")
            continue
        scale_anc = 40.0 / raw_anc_sum
        w_anc_draw = [r * scale_anc for r in raw_anc]
        bs_fl, _ = _weighted_fleiss_kappa_8rater(anchor_8subjs, w_anc_draw)
        if not math.isfinite(bs_fl) or bs_fl < -1.0 or bs_fl > 1.0:
            raise ConceptGraphError("schema sync: non-finite or out-of-bounds anchor bootstrap Fleiss kappa")
        valid_anc += 1

    expected_metrics = {
        "alpha_subject": pt_alpha_subj,
        "kappa_bar_subject": pt_kappa_bar,
        "cond_hooper": pt_hr["cond_hooper"],
        "all_hooper": pt_hr["all_hooper"],
        "cond_rolling": pt_hr["cond_rolling"],
        "all_rolling": pt_hr["all_rolling"],
        "alpha_masi": pt_alpha_masi,
        "empty_rate": pt_hr["empty_rate"],
        "p_max": pt_p_max,
    }
    for m_key, exp_val in expected_metrics.items():
        if abs(float(hs[m_key]) - exp_val) > 1e-9:
            raise ConceptGraphError(
                f"schema sync: human_study_stage2.{m_key}={hs[m_key]} disagrees with raw recomputation ({exp_val:.12f})"
            )
    if hs["min_area_both_non_empty"] != min_area_both_non_empty:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.min_area_both_non_empty={hs['min_area_both_non_empty']} disagrees with recomputed {min_area_both_non_empty}"
        )
    if "leaf_specificity" in hs and (not _is_finite_num(hs["leaf_specificity"]) or abs(float(hs["leaf_specificity"]) - pt_leaf_spec) > 1e-9):
        raise ConceptGraphError("schema sync: human_study_stage2.leaf_specificity disagrees with raw recomputation")
    if "alpha_subject_anchor" in hs and (not _is_finite_num(hs["alpha_subject_anchor"]) or abs(float(hs["alpha_subject_anchor"]) - pt_anc_alpha_subj) > 1e-9):
        raise ConceptGraphError("schema sync: human_study_stage2.alpha_subject_anchor disagrees with raw recomputation")
    if "alpha_masi_anchor" in hs and (not _is_finite_num(hs["alpha_masi_anchor"]) or abs(float(hs["alpha_masi_anchor"]) - pt_anc_alpha_masi) > 1e-9):
        raise ConceptGraphError("schema sync: human_study_stage2.alpha_masi_anchor disagrees with raw recomputation")
    if "kappa_fleiss_anchor" in hs and (not _is_finite_num(hs["kappa_fleiss_anchor"]) or abs(float(hs["kappa_fleiss_anchor"]) - pt_anc_fleiss) > 1e-9):
        raise ConceptGraphError("schema sync: human_study_stage2.kappa_fleiss_anchor disagrees with raw recomputation")
    if hs.get("degenerate_resample_count") != degen_primary:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.degenerate_resample_count={hs.get('degenerate_resample_count')} disagrees with recomputed {degen_primary}"
        )
    if hs.get("anchor_degenerate_resample_count") != degen_anc:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.anchor_degenerate_resample_count={hs.get('anchor_degenerate_resample_count')} disagrees with recomputed {degen_anc}"
        )


def _verify_enrichment_alias_meta(raw_av: object, enriched_av: object, cid: str, ak_clean: str) -> dict:
    if not isinstance(raw_av, dict) or not isinstance(enriched_av, dict):
        raise ConceptGraphError(f"schema sync: missing or malformed alias_metadata enrichment on {cid} alias {ak_clean!r}")
    exp_type = str(raw_av.get("type") or "").strip()
    exp_src = " ".join(str(raw_av.get("source") or "").split())
    exp_snote = " ".join(str(raw_av.get("scope_note") or "").split())
    exp_cunits = [str(u).strip() for u in (raw_av.get("checked_dev_units") or [])]
    exp_vq = " ".join(str(raw_av.get("verbatim_quote") or "").split())
    exp_loc = str(raw_av.get("primary_source_locator") or "").strip()
    if (
        enriched_av.get("type") != exp_type
        or enriched_av.get("source") != exp_src
        or enriched_av.get("scope_note") != exp_snote
        or enriched_av.get("checked_dev_units") != exp_cunits
        or enriched_av.get("verbatim_quote") != exp_vq
        or (exp_loc and enriched_av.get("primary_source_locator") != exp_loc)
        or not _HEX64_RE.match(str(enriched_av.get("primary_source_sha256") or ""))
        or len(str(enriched_av.get("primary_source_locator") or "").strip()) < 3
    ):
        raise ConceptGraphError(
            f"schema sync: alias_metadata enrichment on {cid} alias {ak_clean!r} disagrees with reviewed replayed_patch"
        )
    if raw_av.get("primary_source_sha256") and enriched_av.get("primary_source_sha256") != str(raw_av["primary_source_sha256"]).strip():
        raise ConceptGraphError(
            f"schema sync: alias_metadata primary_source_sha256 on {cid} alias {ak_clean!r} disagrees with replayed_patch"
        )
    if exp_type == "upward-posting":
        if enriched_av.get("dev_lineages_requiring_separation") != raw_av.get("dev_lineages_requiring_separation"):
            raise ConceptGraphError(
                f"schema sync: upward-posting dev_lineages_requiring_separation mismatch on {cid} alias {ak_clean!r}"
            )
    return deepcopy(enriched_av)


def _verify_enrichment_warrant_sources(
    raw_ws: object,
    enriched_ws: object,
    cid: str,
    burned_doc: dict | None,
) -> list[dict]:
    if not isinstance(raw_ws, list) or not isinstance(enriched_ws, list) or len(raw_ws) != len(enriched_ws) or len(raw_ws) < 2:
        raise ConceptGraphError(f"schema sync: warrant_sources enrichment on {cid} disagrees with reviewed replayed_patch")
    hist_units = (burned_doc or {}).get("historical_phase3_units") or {}
    merges = resolve_cluster_merges((burned_doc or {}).get("cluster_merges") or {})
    out: list[dict] = []
    for r_item, e_item in zip(raw_ws, enriched_ws):
        if not isinstance(r_item, dict) or not isinstance(e_item, dict):
            raise ConceptGraphError(f"schema sync: malformed warrant_sources enrichment item on {cid}")
        uid = str(r_item.get("unit_id") or "").strip()
        loc = " ".join(str(r_item.get("locator") or "").split())
        if e_item.get("unit_id") != uid or e_item.get("locator") != loc:
            raise ConceptGraphError(f"schema sync: warrant_sources[{uid!r}] on {cid} disagrees with replayed_patch")
        if uid in hist_units:
            hu = hist_units[uid]
            exp_pf = merges.get(str(hu["provenance_family_id"]), str(hu["provenance_family_id"]))
            if (
                e_item.get("stratum") != hu.get("stratum")
                or e_item.get("provenance_family_id") != exp_pf
                or e_item.get("source_type") != hu.get("source_type")
                or e_item.get("origin") != hu.get("origin")
                or bool(e_item.get("editor_affiliated", False)) != bool(hu.get("editor_affiliated", False))
            ):
                raise ConceptGraphError(
                    f"schema sync: warrant_sources[{uid!r}] on {cid} disagrees with pinned historical_phase3_units metadata"
                )
        out.append(deepcopy(e_item))
    return out


def replay_provenance_step(
    prov_state: dict[str, object],
    pre_doc: dict,
    post_doc: dict,
    entry: dict,
    burned_doc: dict | None = None,
) -> None:
    """Deterministically advance top-level provenance state across one campaign_entry."""
    mode = str(entry.get("mode") or "")
    pending_ids: list[str] = list(prov_state["pending_campaign_ids"])  # type: ignore[arg-type]

    if mode == "guides-apply":
        if pending_ids or bool(prov_state["pending_pre_evaluation_override"]):
            raise ConceptGraphError(
                "schema sync: guides-apply is forbidden while structural pre-evaluation campaigns remain pending"
            )
        return

    if mode == "finalize-structural-evaluation":
        if not pending_ids:
            raise ConceptGraphError("schema sync: finalize-structural-evaluation requires non-empty pending_campaign_ids")
        covered = entry.get("covered_pending_campaign_ids")
        if (
            not isinstance(covered, list)
            or not covered
            or sorted(str(x) for x in covered) != sorted(pending_ids)
        ):
            raise ConceptGraphError(
                "schema sync: finalize-structural-evaluation covered_pending_campaign_ids does not cover all pending campaigns"
            )
        prov_state["pending_campaign_ids"] = []
        prov_state["pending_pre_evaluation_override"] = False
        return

    if mode != "full-3family-reconcile":
        raise ConceptGraphError(f"schema sync: unsupported mode {mode!r} in replay_provenance_step")

    camp_id = str(entry.get("campaign_id") or "").strip()
    if len(camp_id) < 3:
        raise ConceptGraphError("schema sync: full-3family-reconcile entry missing valid campaign_id")

    if entry.get("allow_pre_evaluation_apply") is True:
        if camp_id in pending_ids:
            raise ConceptGraphError(f"schema sync: duplicate pending campaign_id {camp_id!r}")
        pending_ids.append(camp_id)
        prov_state["pending_campaign_ids"] = pending_ids
        prov_state["pending_pre_evaluation_override"] = True
    else:
        if pending_ids:
            raise ConceptGraphError(
                "schema sync: non-staged full-3family-reconcile is forbidden while pending_campaign_ids remain unfinalized"
            )
        prov_state["pending_campaign_ids"] = []
        prov_state["pending_pre_evaluation_override"] = False

    change = entry["replayed_patch"]
    enrichment = entry.get("provenance_evidence_enrichment")
    if not isinstance(enrichment, dict):
        raise ConceptGraphError("schema sync: full-3family-reconcile entry missing provenance_evidence_enrichment mapping")
    ws_enrich = enrichment.get("warrant_sources") or {}
    am_enrich = enrichment.get("alias_metadata") or {}
    if not isinstance(ws_enrich, dict) or not isinstance(am_enrich, dict):
        raise ConceptGraphError("schema sync: malformed provenance_evidence_enrichment sub-mappings")

    prov_concepts: dict[str, dict] = prov_state["concepts"]  # type: ignore[assignment]
    prov_subjects: dict[str, dict] = prov_state["subjects"]  # type: ignore[assignment]
    prov_retired: dict[str, dict] = prov_state["retired_concepts"]  # type: ignore[assignment]
    prov_gf: dict[str, dict] = prov_state["grandfathered_draft_alias_exceptions"]  # type: ignore[assignment]

    rows = {row["id"]: row for area in pre_doc["areas"] for row in area["rows"]}
    code2row = {row["code"]: row["id"] for row in rows.values() if "code" in row}
    pre_concepts = {c["id"]: c for c in pre_doc["concepts"]}
    post_concepts = {c["id"]: c for c in post_doc["concepts"]}

    def resolve_row_id(home_ref: str) -> str:
        return code2row.get(home_ref, home_ref)

    expected_ws_cids: set[str] = set()
    expected_am_cids: set[str] = set()
    removed_rel_ids = {str(x).strip() for x in (change.get("relationships_remove") or []) if str(x).strip()}

    for u in change.get("concepts_update", []) or []:
        cid = u["id"].strip()
        post_node = post_concepts.get(cid) or pre_concepts[cid]
        cprov = deepcopy(prov_concepts.get(cid) or {})
        existing_ameta = dict(cprov.get("alias_metadata") or {})

        rem_list = [str(x).strip() for x in u.get("remove_aliases", []) or []]
        rem_norms = {_norm_prov_tokens(x) for x in rem_list}
        if rem_norms:
            existing_ameta = {ak: av for ak, av in existing_ameta.items() if _norm_prov_tokens(ak) not in rem_norms}
            if isinstance(cprov.get("restored_aliases"), dict):
                cleaned_rest = {
                    rk: rv for rk, rv in cprov["restored_aliases"].items() if _norm_prov_tokens(rk) not in rem_norms
                }
                if cleaned_rest:
                    cprov["restored_aliases"] = cleaned_rest
                else:
                    cprov.pop("restored_aliases", None)
            rat_map_in = u.get("remove_aliases_rationales") or {}
            rat_single = " ".join(str(u.get("remove_aliases_rationale") or "").split())
            rem_map = dict(cprov.get("removed_aliases") or {})
            for ra in rem_list:
                per_a_rat = ""
                if isinstance(rat_map_in, dict):
                    for rk_in, rv_in in rat_map_in.items():
                        if _norm_prov_tokens(str(rk_in)) == _norm_prov_tokens(ra):
                            per_a_rat = " ".join(str(rv_in).split())
                            break
                chosen_rat = per_a_rat or rat_single
                if len(chosen_rat) < 15:
                    raise ConceptGraphError(f"schema sync: missing >=15-char removal rationale for alias {ra!r} on {cid}")
                rem_map[ra] = chosen_rat
            cprov["removed_aliases"] = rem_map
            if cid in prov_gf:
                rem_gf = [ga for ga in prov_gf[cid]["aliases"] if _norm_prov_tokens(ga) not in rem_norms]
                if rem_gf:
                    prov_gf[cid] = {**prov_gf[cid], "aliases": rem_gf}
                else:
                    prov_gf.pop(cid, None)

        added_alist = [str(x).strip() for x in (u.get("add_aliases") or [])]
        attested_alist = [str(x).strip() for x in (u.get("attest_existing_aliases") or [])]
        meta_targets = added_alist + attested_alist
        raw_am_u = u.get("alias_metadata")
        if meta_targets or raw_am_u:
            if not isinstance(raw_am_u, dict) or {str(k).strip() for k in raw_am_u} != set(meta_targets):
                raise ConceptGraphError(f"schema sync: alias_metadata keys on {cid} disagree with add_aliases/attest_existing_aliases")
            expected_am_cids.add(cid)
            enr_u = am_enrich.get(cid)
            if not isinstance(enr_u, dict) or set(enr_u.keys()) != set(meta_targets):
                raise ConceptGraphError(f"schema sync: provenance_evidence_enrichment alias_metadata mismatch on {cid}")
            raw_am_clean = {str(k).strip(): v for k, v in raw_am_u.items()}
            existing_rem_norms = {_norm_prov_tokens(rk): rk for rk in (cprov.get("removed_aliases") or {})}
            rest_map = dict(cprov.get("restored_aliases") or {})
            for ak_clean in meta_targets:
                existing_ameta[ak_clean] = _verify_enrichment_alias_meta(raw_am_clean[ak_clean], enr_u[ak_clean], cid, ak_clean)
            for ak_clean in added_alist:
                ak_norm = _norm_prov_tokens(ak_clean)
                if ak_norm in existing_rem_norms:
                    if u.get("restore_removed_alias") is not True:
                        raise ConceptGraphError(
                            f"schema sync: restoring tombstoned alias {ak_clean!r} on {cid} requires restore_removed_alias=true"
                        )
                    rest_rat = " ".join(str(u.get("restore_aliases_rationale") or "").split())
                    if len(rest_rat) < 15:
                        raise ConceptGraphError(f"schema sync: restore_aliases_rationale on {cid} must be >=15 chars")
                    rest_map = {rk: rv for rk, rv in rest_map.items() if _norm_prov_tokens(rk) != ak_norm}
                    orig_tomb_key = existing_rem_norms[ak_norm]
                    rest_map[orig_tomb_key] = rest_rat
                    if ak_clean != orig_tomb_key:
                        rest_map[ak_clean] = rest_rat
            for ak_clean in attested_alist:
                ak_norm = _norm_prov_tokens(ak_clean)
                if cid in prov_gf:
                    rem_gf = [ga for ga in prov_gf[cid]["aliases"] if _norm_prov_tokens(ga) != ak_norm]
                    if rem_gf:
                        prov_gf[cid] = {**prov_gf[cid], "aliases": rem_gf}
                    else:
                        prov_gf.pop(cid, None)
            if rest_map:
                cprov["restored_aliases"] = rest_map

        if existing_ameta:
            cprov["alias_metadata"] = existing_ameta
        else:
            cprov.pop("alias_metadata", None)
        if u.get("definition"):
            cprov["definition_change_kind"] = str(u["definition_change_kind"]).strip()
            cprov["edge_impact_note"] = " ".join(str(u["edge_impact_note"]).split())
            raw_checked_inc = u.get("checked_incident_relationships")
            if not isinstance(raw_checked_inc, list):
                raise ConceptGraphError(
                    f"schema sync: concepts_update[{cid}] definition edit requires checked_incident_relationships list"
                )
            clean_inc = [str(x).strip() for x in raw_checked_inc if isinstance(x, str) and x.strip()]
            expected_inc = sorted(
                {
                    str(e["id"])
                    for e in pre_doc.get("relationships", [])
                    if str(e["id"]) not in removed_rel_ids and (e.get("source") == cid or e.get("target") == cid)
                }
            )
            if len(clean_inc) != len(raw_checked_inc) or len(set(clean_inc)) != len(clean_inc) or sorted(clean_inc) != expected_inc:
                raise ConceptGraphError(
                    f"schema sync: checked_incident_relationships on {cid} disagrees with surviving baseline incident relationships {expected_inc}"
                )
            cprov["checked_incident_relationships"] = sorted(clean_inc)
        if u.get("label"):
            cprov["updated_label"] = " ".join(str(u["label"]).split())
        if u.get("scope"):
            cprov["updated_scope"] = str(u["scope"]).strip()
        if cprov:
            prov_concepts[cid] = cprov
        else:
            prov_concepts.pop(cid, None)

    for mv in change.get("moves", []) or []:
        mcid = mv["id"].strip()
        mhome = resolve_row_id(mv["home"].strip())
        if mhome in prov_subjects and "secondary_concepts_rationale" in prov_subjects[mhome]:
            prov_subjects[mhome]["secondary_concepts_rationale"].pop(mcid, None)
            if not prov_subjects[mhome]["secondary_concepts_rationale"]:
                prov_subjects.pop(mhome, None)

    target_version = str(change.get("version") or pre_doc["version"]).strip()
    for c in change.get("concepts_add", []) or []:
        cid = c["id"].strip()
        expected_ws_cids.add(cid)
        cprov_new: dict[str, object] = {
            "added_in": target_version,
            "warrant_sources": _verify_enrichment_warrant_sources(c.get("warrant_sources"), ws_enrich.get(cid), cid, burned_doc),
            "witness_justification": deepcopy(c["witness_justification"]),
        }
        c_aliases = [str(x).strip() for x in (c.get("aliases") or [])]
        raw_cam = c.get("alias_metadata")
        if c_aliases or raw_cam:
            if not isinstance(raw_cam, dict) or {str(k).strip() for k in raw_cam} != set(c_aliases):
                raise ConceptGraphError(f"schema sync: concepts_add alias_metadata mismatch on {cid}")
            expected_am_cids.add(cid)
            enr_c = am_enrich.get(cid)
            if not isinstance(enr_c, dict) or set(enr_c.keys()) != set(c_aliases):
                raise ConceptGraphError(f"schema sync: provenance_evidence_enrichment concepts_add alias_metadata mismatch on {cid}")
            raw_cam_clean = {str(k).strip(): v for k, v in raw_cam.items()}
            cprov_new["alias_metadata"] = {
                ak_c: _verify_enrichment_alias_meta(raw_cam_clean[ak_c], enr_c[ak_c], cid, ak_c) for ak_c in c_aliases
            }
        prov_concepts[cid] = cprov_new

    if set(ws_enrich.keys()) != expected_ws_cids or set(am_enrich.keys()) != expected_am_cids:
        raise ConceptGraphError("schema sync: provenance_evidence_enrichment contains extraneous concept keys")

    # Resolve retirements using pre-step concept state (after concepts_update/moves/concepts_add).
    mid_concepts = dict(pre_concepts)
    for c_added_node in post_doc.get("concepts", []):
        if c_added_node["id"] not in mid_concepts:
            mid_concepts[c_added_node["id"]] = c_added_node
    for u in change.get("concepts_update", []) or []:
        ucid = u["id"].strip()
        if ucid in mid_concepts:
            tmp_node = deepcopy(mid_concepts[ucid])
            if u.get("label"):
                tmp_node["label"] = " ".join(str(u["label"]).split())
            rem_n = {_norm_prov_tokens(str(x)) for x in (u.get("remove_aliases") or [])}
            als = [str(a) for a in (tmp_node.get("aliases") or []) if _norm_prov_tokens(str(a)) not in rem_n]
            als.extend(str(a).strip() for a in (u.get("add_aliases") or []))
            if als:
                tmp_node["aliases"] = als
            else:
                tmp_node.pop("aliases", None)
            mid_concepts[ucid] = tmp_node

    for r in change.get("concepts_retire", []) or []:
        rcid = r["id"].strip()
        rep = r["replaced_by"].strip()
        former = mid_concepts.pop(rcid)
        prov_gf.pop(rcid, None)
        former_lbl = str(former.get("label") or "").strip()
        former_als = [str(a) for a in (former.get("aliases") or [])]
        for rid_k in list(prov_subjects.keys()):
            if "secondary_concepts_rationale" in prov_subjects[rid_k]:
                prov_subjects[rid_k]["secondary_concepts_rationale"].pop(rcid, None)
                if not prov_subjects[rid_k]["secondary_concepts_rationale"]:
                    prov_subjects.pop(rid_k, None)
        old_cprov = prov_concepts.pop(rcid, None) or {}
        prov_retired[rcid] = {
            **old_cprov,
            "former_label": former_lbl,
            "former_aliases": former_als,
            "replaced_by": rep,
            "retired_in": str(r.get("version") or target_version).strip(),
            "reason": " ".join(str(r["reason"]).split()),
        }

    if post_doc.get("retired"):
        rep_map = {rec_r["id"]: rec_r["replaced_by"] for rec_r in post_doc["retired"]}
        for rcid_k, rprov_v in prov_retired.items():
            if rcid_k in rep_map:
                rprov_v["replaced_by"] = rep_map[rcid_k]

    for su in change.get("subject_updates", []) or []:
        rid = resolve_row_id(su["row"].strip())
        if "remove_secondary_concepts" in su or "add_secondary_concepts" in su:
            rem_secs = [s.strip() for s in su.get("remove_secondary_concepts", []) or []]
            sprov = deepcopy(prov_subjects.get(rid) or {})
            sec_rat_map = dict(sprov.get("secondary_concepts_rationale") or {})
            for rs in rem_secs:
                sec_rat_map.pop(rs, None)
            rat_text = " ".join(str(su.get("rationale") or "").split())
            for raw_s in su.get("add_secondary_concepts", []) or []:
                s = raw_s.strip()
                if len(rat_text) < 15:
                    raise ConceptGraphError(f"schema sync: missing >=15-char rationale for secondary_concept {s} on {rid}")
                sec_rat_map[s] = rat_text
            if sec_rat_map:
                sprov["secondary_concepts_rationale"] = sec_rat_map
                prov_subjects[rid] = sprov
            else:
                prov_subjects.pop(rid, None)

    if entry.get("release_governance_receipt") is not None:
        prov_state["release_governance_receipt"] = deepcopy(entry["release_governance_receipt"])
    prov_state["last_reconcile_receipt"] = entry


def build_provenance_doc_from_state(
    prov_state: dict[str, object],
    *,
    taxonomy_version: str,
    taxonomy_sha256: str,
    non_guides_taxonomy_sha256: str,
    applied_campaigns: list[dict],
) -> dict[str, object]:
    """Serialize a canonical term-provenance.yml mapping with self-hash from replayed state."""
    prov_concepts: dict[str, dict] = prov_state["concepts"]  # type: ignore[assignment]
    prov_subjects: dict[str, dict] = prov_state["subjects"]  # type: ignore[assignment]
    prov_retired: dict[str, dict] = prov_state["retired_concepts"]  # type: ignore[assignment]
    prov_gf: dict[str, dict] = prov_state["grandfathered_draft_alias_exceptions"]  # type: ignore[assignment]

    prov_doc: dict[str, object] = {
        "schema_version": 2,
        "taxonomy_version": str(taxonomy_version),
        "taxonomy_sha256": str(taxonomy_sha256),
        "non_guides_taxonomy_sha256": str(non_guides_taxonomy_sha256),
        "pending_pre_evaluation_override": bool(prov_state["pending_pre_evaluation_override"]),
        "pending_campaign_ids": list(prov_state["pending_campaign_ids"]),  # type: ignore[arg-type]
        "last_reconcile_receipt": prov_state["last_reconcile_receipt"],
        "applied_campaigns": list(applied_campaigns),
        "concepts": {k: prov_concepts[k] for k in sorted(prov_concepts) if prov_concepts[k]},
        "subjects": {k: prov_subjects[k] for k in sorted(prov_subjects) if prov_subjects[k]},
    }
    if prov_retired:
        prov_doc["retired_concepts"] = {k: prov_retired[k] for k in sorted(prov_retired)}
    if prov_gf:
        prov_doc["grandfathered_draft_alias_exceptions"] = {k: prov_gf[k] for k in sorted(prov_gf)}
    if prov_state.get("release_governance_receipt") is not None:
        prov_doc["release_governance_receipt"] = deepcopy(prov_state["release_governance_receipt"])
    prov_doc["sidecar_content_sha256"] = compute_sidecar_content_sha256(prov_doc)
    return prov_doc


def validate_term_provenance_sidecar(root: Path, *, verify_live_script_digests: bool = False) -> None:
    """Verify hash-chained provenance, deterministic genesis replay, and Gate 2/5 records in term-provenance.yml.

    Security & integrity boundary: `sidecar_content_sha256`, `campaign_entry_sha256`, canonical patch digests,
    and deterministic genesis replay of both taxonomy and top-level provenance metadata detect un-reconciled
    YAML edits, partial writes, provenance tampering, or untracked edits to 0.4-draft concepts. External
    authorization against a committer modifying validator code is supplied by git history and CI diff review
    against `origin/dev`.
    """
    from .field_graph import load_taxonomy_yaml

    prov_path = root / "taxonomy/term-provenance.yml"
    tax_path = root / "taxonomy/ml-systems-taxonomy.yml"
    if not prov_path.exists():
        raise ConceptGraphError("schema sync: taxonomy/term-provenance.yml is missing (run reconcile.py and apply.py)")
    if not tax_path.exists():
        raise ConceptGraphError("schema sync: taxonomy/ml-systems-taxonomy.yml is missing")

    prov_text = prov_path.read_text()
    if ("/" + "Users/") in prov_text or ("/" + "home/") in prov_text:
        raise ConceptGraphError("schema sync: machine-local path detected in taxonomy/term-provenance.yml")

    prov = load_taxonomy_yaml(prov_path)
    raw_tax = load_taxonomy_yaml(tax_path)
    if not isinstance(prov, dict) or prov.get("schema_version") != 2:
        raise ConceptGraphError("schema sync: taxonomy/term-provenance.yml must be a schema_version=2 mapping")
    if not isinstance(raw_tax, dict):
        raise ConceptGraphError("schema sync: taxonomy/ml-systems-taxonomy.yml must be a mapping")

    expected_tax_sha = hashlib.sha256(tax_path.read_bytes()).hexdigest()
    if prov.get("taxonomy_sha256") != expected_tax_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/term-provenance.yml taxonomy_sha256 disagrees with taxonomy/ml-systems-taxonomy.yml "
            "(re-run reconcile.py and apply.py)"
        )
    expected_non_guides_sha = compute_non_guides_taxonomy_sha256(raw_tax)
    if prov.get("non_guides_taxonomy_sha256") != expected_non_guides_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/term-provenance.yml non_guides_taxonomy_sha256 disagrees with taxonomy/ml-systems-taxonomy.yml"
        )
    if prov.get("taxonomy_version") != str(raw_tax.get("version")):
        raise ConceptGraphError("schema sync: taxonomy/term-provenance.yml taxonomy_version disagrees with ml-systems-taxonomy.yml")

    expected_sidecar_sha = compute_sidecar_content_sha256(prov)
    if prov.get("sidecar_content_sha256") != expected_sidecar_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/term-provenance.yml sidecar_content_sha256 mismatch (hand-edited outside apply.py)"
        )

    campaigns = prov.get("applied_campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ConceptGraphError("schema sync: taxonomy/term-provenance.yml must record non-empty applied_campaigns hash chain")

    if isinstance(campaigns[-1], dict):
        if campaigns[-1].get("post_apply_taxonomy_sha256") != expected_tax_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_apply_taxonomy_sha256 disagrees with taxonomy_sha256")
        if campaigns[-1].get("post_non_guides_taxonomy_sha256") != expected_non_guides_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_non_guides_taxonomy_sha256 disagrees with non_guides_taxonomy_sha256")

    genesis_doc = load_genesis_baseline_taxonomy(root)
    genesis_cids = {c["id"] for c in genesis_doc.get("concepts", [])}
    burned_doc = _load_burned_clusters(root)
    replayed_doc = deepcopy(genesis_doc)
    replayed_prov_state = init_genesis_provenance_state(genesis_doc)
    reconstructed_step_sidecar_shas: list[str] = []

    governed_added_cids: set[str] = set()
    governed_touched_cids: set[str] = set()
    governed_retired_cids: set[str] = set()
    governed_def_updated_cids: set[str] = set()
    net_aliases_added: dict[str, set[str]] = {}
    all_aliases_removed: dict[str, set[str]] = {}
    restored_alias_norms_by_cid: dict[str, set[str]] = {}
    global_tombstoned_norms: dict[str, set[str]] = {}
    net_sec_added: dict[str, set[str]] = {}
    last_full_rec: dict | None = None
    seen_campaign_ids: set[str] = set()
    version_setting_entry: dict | None = None

    for idx, entry in enumerate(campaigns):
        if not isinstance(entry, dict):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must be a mapping")
        mode = entry.get("mode")
        base_sha = str(entry.get("baseline_taxonomy_sha256") or "")
        post_sha = str(entry.get("post_apply_taxonomy_sha256") or "")
        base_ng_sha = str(entry.get("baseline_non_guides_taxonomy_sha256") or "")
        post_ng_sha = str(entry.get("post_non_guides_taxonomy_sha256") or "")
        if (
            not _HEX64_RE.match(base_sha)
            or not _HEX64_RE.match(post_sha)
            or not _HEX64_RE.match(base_ng_sha)
            or not _HEX64_RE.match(post_ng_sha)
        ):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing valid baseline/post taxonomy or non-guides SHA-256")

        entry_sha = str(entry.get("campaign_entry_sha256") or "")
        if not _HEX64_RE.match(entry_sha) or entry_sha != compute_campaign_entry_sha256(entry):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] campaign_entry_sha256 mismatch")

        base_prov_sha = entry.get("baseline_term_provenance_sha256")
        prior_sidecar_sha = entry.get("prior_sidecar_content_sha256")
        prev_entry_sha = entry.get("previous_campaign_entry_sha256")
        if idx == 0:
            if mode != "full-3family-reconcile" or entry.get("bootstrap_provenance") is not True:
                raise ConceptGraphError("schema sync: applied_campaigns[0] must be a bootstrap full-3family-reconcile entry")
            if base_sha != GENESIS_PRE_SIDECAR_TAXONOMY_SHA256 or base_ng_sha != GENESIS_PRE_SIDECAR_NON_GUIDES_SHA256:
                raise ConceptGraphError(
                    "schema sync: applied_campaigns[0] baseline_taxonomy_sha256 does not match pinned genesis 0.4-draft taxonomy"
                )
            if base_prov_sha is not None or prior_sidecar_sha is not None or prev_entry_sha is not None:
                raise ConceptGraphError("schema sync: bootstrap applied_campaigns[0] must have null baseline_term_provenance_sha256 and prior_sidecar_content_sha256")
        else:
            if bool(entry.get("bootstrap_provenance")):
                raise ConceptGraphError(f"schema sync: bootstrap_provenance forbidden after genesis at applied_campaigns[{idx}]")
            if base_sha != campaigns[idx - 1].get("post_apply_taxonomy_sha256"):
                raise ConceptGraphError(f"schema sync: broken taxonomy SHA-256 chain at applied_campaigns[{idx}]")
            if base_ng_sha != campaigns[idx - 1].get("post_non_guides_taxonomy_sha256"):
                raise ConceptGraphError(f"schema sync: broken non_guides_taxonomy_sha256 chain at applied_campaigns[{idx}]")
            if not isinstance(base_prov_sha, str) or not _HEX64_RE.match(base_prov_sha):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] requires valid baseline_term_provenance_sha256")
            if not isinstance(prior_sidecar_sha, str) or not _HEX64_RE.match(prior_sidecar_sha):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] requires valid prior_sidecar_content_sha256")
            if prior_sidecar_sha != reconstructed_step_sidecar_shas[idx - 1]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] prior_sidecar_content_sha256 disagrees with replayed step {idx - 1} sidecar digest"
                )
            if prev_entry_sha != campaigns[idx - 1].get("campaign_entry_sha256"):
                raise ConceptGraphError(f"schema sync: broken previous_campaign_entry_sha256 chain at applied_campaigns[{idx}]")

        replayed_patch = entry.get("replayed_patch")
        if not isinstance(replayed_patch, dict) or not replayed_patch:
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing non-empty replayed_patch")

        pre_step_doc = deepcopy(replayed_doc)

        if mode == "full-3family-reconcile":
            if entry.get("receipt_schema_version") != 4:
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must have receipt_schema_version=4")
            camp_id = str(entry.get("campaign_id") or "").strip()
            if len(camp_id) < 3 or camp_id in seen_campaign_ids:
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing or duplicate campaign_id {camp_id!r}")
            seen_campaign_ids.add(camp_id)

            if entry.get("allow_pre_evaluation_apply") is True and replayed_patch.get("version"):
                raise ConceptGraphError(
                    f"schema sync: allow_pre_evaluation_apply is forbidden with a version bump at applied_campaigns[{idx}]"
                )
            for hkey in (
                "reconcile_script_sha256",
                "apply_script_sha256",
                "taxonomy_schema_script_sha256",
                "run_reviews_script_sha256",
                "stage_reviewer_workspace_script_sha256",
                "baseline_linkml_schema_sha256",
                "burned_clusters_sha256",
                "sampling_manifest_sha256",
                "workspace_isolation_r2_sha256",
                "run_reviews_r2_sha256",
                "patch_yaml_sha256",
                "final_yaml_sha256",
                "final_yaml_canonical_json_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing valid {hkey}")
            if canonical_patch_json_sha256(replayed_patch) != entry["final_yaml_canonical_json_sha256"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] replayed_patch canonical JSON SHA-256 disagrees with final_yaml_canonical_json_sha256"
                )
            r2_shas = entry.get("r2_reviews_sha256")
            if (
                not isinstance(r2_shas, dict)
                or set(r2_shas.keys()) != {"claude-r2.yaml", "codex-r2.yaml", "gemini-r2.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in r2_shas.values())
            ):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must bind all 3 *-r2.yaml hashes")
            ops = entry.get("operations_applied")
            if not isinstance(ops, list) or not ops or not all(isinstance(x, str) and x for x in ops):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must list non-empty operations_applied")
            for op in ops:
                if op != "version" and not op.startswith(_ALLOWED_OP_PREFIXES):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] unrecognized operation prefix {op!r}")

            details = entry.get("operation_details")
            if not isinstance(details, dict):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing structured operation_details")

            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if sorted(ops) != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] operations_applied disagrees with replayed_patch"
                )
            if compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] replayed non_guides_taxonomy_sha256 disagrees with post_non_guides_taxonomy_sha256"
                )
            if replayed_patch.get("version"):
                version_setting_entry = entry

            c_added = sorted(details.get("concepts_added") or [])
            c_retired = sorted(details.get("concepts_retired") or [])
            d_upd = sorted(details.get("definitions_updated") or [])
            a_added_map = {k: list(v) for k, v in sorted((details.get("aliases_added") or {}).items())}
            a_att_map = {k: list(v) for k, v in sorted((details.get("aliases_attested") or {}).items())}
            a_rem_map = {k: list(v) for k, v in sorted((details.get("aliases_removed") or {}).items())}
            a_rest_map = details.get("aliases_restored") or {}
            other_c_upd = sorted(details.get("other_concept_updates") or [])
            s_add_map = {k: list(v) for k, v in sorted((details.get("secondary_concepts_added") or {}).items())}
            s_rem_map = {k: list(v) for k, v in sorted((details.get("secondary_concepts_removed") or {}).items())}
            other_s_upd = sorted(details.get("other_subject_updates") or [])

            if (
                c_added != replay_meta["concepts_added"]
                or c_retired != replay_meta["concepts_retired"]
                or d_upd != replay_meta["definitions_updated"]
                or a_added_map != replay_meta["aliases_added"]
                or a_att_map != replay_meta["aliases_attested"]
                or a_rem_map != replay_meta["aliases_removed"]
                or other_c_upd != replay_meta["other_concept_updates"]
                or s_add_map != replay_meta["secondary_concepts_added"]
                or s_rem_map != replay_meta["secondary_concepts_removed"]
                or other_s_upd != replay_meta["other_subject_updates"]
                or bool(details.get("guides_replaced", False)) != replay_meta["guides_replaced"]
            ):
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] operation_details disagrees with replayed_patch"
                )

            for rcid_rest, rlist in a_rest_map.items():
                if not isinstance(rlist, list) or not rlist or not set(rlist).issubset(set(a_added_map.get(rcid_rest) or [])):
                    raise ConceptGraphError(
                        f"schema sync: applied_campaigns[{idx}] aliases_restored on {rcid_rest} must be a non-empty subset of aliases_added"
                    )

            for cid in c_added:
                governed_added_cids.add(cid)
                governed_touched_cids.add(cid)
            for cid in set(d_upd) | set(a_added_map) | set(a_att_map) | set(a_rem_map):
                governed_touched_cids.add(cid)
            for rcid in c_retired:
                governed_retired_cids.add(rcid)
                governed_added_cids.discard(rcid)
                governed_def_updated_cids.discard(rcid)
                net_aliases_added.pop(rcid, None)

            governed_def_updated_cids.update(d_upd)
            for cid, rlist in a_rem_map.items():
                all_aliases_removed.setdefault(cid, set()).update(rlist)
                rem_norms_step = {_norm_prov_tokens(ra) for ra in rlist}
                for rn in rem_norms_step:
                    global_tombstoned_norms.setdefault(rn, set()).add(cid)
                if cid in net_aliases_added:
                    net_aliases_added[cid] = {
                        ea for ea in net_aliases_added[cid] if _norm_prov_tokens(ea) not in rem_norms_step
                    }
                if cid in restored_alias_norms_by_cid:
                    restored_alias_norms_by_cid[cid].difference_update(rem_norms_step)
            for cid, alist in a_added_map.items():
                net_aliases_added.setdefault(cid, set()).update(alist)
            for cid, alist in a_att_map.items():
                net_aliases_added.setdefault(cid, set()).update(alist)
            for cid, rlist in a_rest_map.items():
                for ra in rlist:
                    rn = _norm_prov_tokens(ra)
                    restored_alias_norms_by_cid.setdefault(cid, set()).add(rn)
                    if rn in global_tombstoned_norms:
                        global_tombstoned_norms[rn].discard(cid)
            for rid, slist in s_add_map.items():
                net_sec_added.setdefault(rid, set()).update(slist)
            for rid, slist in s_rem_map.items():
                if rid in net_sec_added:
                    net_sec_added[rid].difference_update(slist)

            last_full_rec = entry
        elif mode == "guides-apply":
            for hkey in (
                "guides_patch_yaml_sha256",
                "guides_final_yaml_sha256",
                "guides_final_yaml_canonical_json_sha256",
                "guides_reconcile_receipt_sha256",
                "guides_apply_script_sha256",
                "apply_script_sha256",
                "reconcile_script_sha256",
                "taxonomy_schema_script_sha256",
                "workspace_isolation_guides_sha256",
                "run_reviews_g1_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply missing valid {hkey}")
            if canonical_patch_json_sha256(replayed_patch) != entry["guides_final_yaml_canonical_json_sha256"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides replayed_patch canonical JSON SHA-256 mismatch"
                )
            g1_shas = entry.get("g1_reviews_sha256")
            if (
                not isinstance(g1_shas, dict)
                or set(g1_shas.keys()) != {"claude-g1.yaml", "codex-g1.yaml", "gemini-g1.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in g1_shas.values())
            ):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply must bind all 3 *-g1.yaml hashes")
            gops = entry.get("guides_operations_applied")
            if not isinstance(gops, list) or not gops or not all(isinstance(x, str) and x for x in gops):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply must list non-empty guides_operations_applied")
            for gop in gops:
                if not gop.startswith(_ALLOWED_GUIDES_OP_PREFIXES):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] unrecognized guides operation {gop!r}")

            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if sorted(gops) != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides_operations_applied disagrees with replayed_patch"
                )
            if post_ng_sha != base_ng_sha or compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides-apply modified non-guides taxonomy content"
                )
        elif mode == "finalize-structural-evaluation":
            for hkey in (
                "apply_script_sha256",
                "reconcile_script_sha256",
                "taxonomy_schema_script_sha256",
                "burned_clusters_sha256",
                "unseal_holdout_receipt_sha256",
                "evaluation_receipt_sha256",
                "audit_receipt_sha256",
                "workspace_isolation_a1_sha256",
                "run_reviews_a1_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(
                        f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation missing valid {hkey}"
                    )
            a1_shas = entry.get("a1_reviews_sha256")
            if (
                not isinstance(a1_shas, dict)
                or set(a1_shas.keys()) != {"claude-a1.yaml", "codex-a1.yaml", "gemini-a1.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in a1_shas.values())
            ):
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation must bind all 3 *-a1.yaml hashes"
                )
            if entry.get("evaluation_gate_completed") is not True:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation requires evaluation_gate_completed=true"
                )
            if post_sha != base_sha or post_ng_sha != base_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation must not mutate taxonomy YAML"
                )
            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if entry.get("operations_applied") != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation operations_applied mismatch"
                )
        else:
            raise ConceptGraphError(f"schema sync: unsupported applied_campaigns mode {mode!r}")

        replay_provenance_step(replayed_prov_state, pre_step_doc, replayed_doc, entry, burned_doc=burned_doc)
        step_prov_doc = build_provenance_doc_from_state(
            replayed_prov_state,
            taxonomy_version=str(replayed_doc["version"]),
            taxonomy_sha256=post_sha,
            non_guides_taxonomy_sha256=post_ng_sha,
            applied_campaigns=campaigns[: idx + 1],
        )
        reconstructed_step_sidecar_shas.append(str(step_prov_doc["sidecar_content_sha256"]))

    if verify_live_script_digests and isinstance(campaigns[-1], dict):
        live_ts = root / "binder/cli/core/taxonomy_schema.py"
        if live_ts.exists() and campaigns[-1].get("taxonomy_schema_script_sha256") != hashlib.sha256(live_ts.read_bytes()).hexdigest():
            raise ConceptGraphError("schema sync: latest campaign taxonomy_schema_script_sha256 disagrees with live taxonomy_schema.py")

    if last_full_rec is None or prov.get("last_reconcile_receipt") != last_full_rec:
        raise ConceptGraphError("schema sync: last_reconcile_receipt disagrees with latest full-3family-reconcile campaign entry")

    if canonical_taxonomy_json(replayed_doc) != canonical_taxonomy_json(raw_tax):
        raise ConceptGraphError(
            "schema sync: replaying applied_campaigns from genesis 0.4-draft does not reproduce ml-systems-taxonomy.yml "
            "(untracked edit outside governed patches)"
        )

    expected_final_prov = build_provenance_doc_from_state(
        replayed_prov_state,
        taxonomy_version=str(raw_tax["version"]),
        taxonomy_sha256=expected_tax_sha,
        non_guides_taxonomy_sha256=expected_non_guides_sha,
        applied_campaigns=campaigns,
    )
    if canonical_taxonomy_json(expected_final_prov) != canonical_taxonomy_json(prov):
        raise ConceptGraphError(
            "schema sync: replaying applied_campaigns provenance state from genesis does not match term-provenance.yml "
            "(unreconciled top-level provenance edit detected)"
        )

    unclosed_pre_eval = bool(replayed_prov_state["pending_pre_evaluation_override"])
    pending_ids_final = list(replayed_prov_state["pending_campaign_ids"])  # type: ignore[arg-type]
    tax_ver_str = str(raw_tax.get("version") or "")
    is_non_draft = not tax_ver_str.endswith("-draft")

    if unclosed_pre_eval and (is_non_draft or prov.get("release_candidate_ready") is True):
        raise ConceptGraphError(
            "schema sync: unclosed allow_pre_evaluation_apply override cannot ship in a release candidate or final release"
        )
    if bool(prov.get("pending_pre_evaluation_override", False)) != unclosed_pre_eval:
        raise ConceptGraphError(
            "schema sync: pending_pre_evaluation_override flag disagrees with applied_campaigns pre-evaluation state"
        )
    if (prov.get("pending_campaign_ids") or []) != pending_ids_final:
        raise ConceptGraphError(
            "schema sync: pending_campaign_ids disagrees with applied_campaigns pre-evaluation state"
        )

    if is_non_draft:
        if prov.get("grandfathered_draft_alias_exceptions"):
            raise ConceptGraphError(
                f"schema sync: non-draft release {tax_ver_str!r} cannot retain unresolved grandfathered_draft_alias_exceptions"
            )
        if (
            version_setting_entry is None
            or version_setting_entry.get("mode") != "full-3family-reconcile"
            or version_setting_entry.get("evaluation_gate_completed") is not True
        ):
            raise ConceptGraphError(
                f"schema sync: non-draft release {tax_ver_str!r} requires a full-3family-reconcile version transition with evaluation_gate_completed=true"
            )
    verify_release_governance_receipt(
        prov.get("release_governance_receipt"),
        tax_ver_str,
        expected_tax_sha,
        expected_non_guides_sha,
        root=root,
        candidate_tax_doc=raw_tax,
        burned_doc=burned_doc,
        prov_concepts=prov.get("concepts") if isinstance(prov.get("concepts"), dict) else None,
    )

    concepts_by_id = {c["id"]: c for c in raw_tax.get("concepts", [])}
    retired_by_id = {r["id"]: r for r in raw_tax.get("retired", []) or []}
    rows_by_id = {r["id"]: r for a in raw_tax.get("areas", []) for r in a.get("rows", [])}

    prov_concepts = prov.get("concepts")
    prov_subjects = prov.get("subjects")
    prov_retired = prov.get("retired_concepts") or {}
    if not isinstance(prov_concepts, dict) or not isinstance(prov_subjects, dict) or not isinstance(prov_retired, dict):
        raise ConceptGraphError("schema sync: term-provenance concepts, subjects, and retired_concepts must be mappings")

    for cid in sorted(governed_touched_cids):
        if cid in concepts_by_id and cid not in prov_concepts:
            raise ConceptGraphError(f"schema sync: governed concept {cid} missing provenance record in term-provenance.yml")
    for rcid in sorted(governed_retired_cids):
        if rcid not in retired_by_id or rcid not in prov_retired:
            raise ConceptGraphError(f"schema sync: retired concept {rcid} missing from retired_concepts in term-provenance.yml")

    retired_reserved_norms: dict[str, str] = {}
    for rcid, rprov in prov_retired.items():
        if (
            rcid not in retired_by_id
            or not isinstance(rprov, dict)
            or len(str(rprov.get("former_label") or "").strip()) < 2
            or not isinstance(rprov.get("former_aliases"), list)
            or rprov.get("replaced_by") not in concepts_by_id
            or len(str(rprov.get("reason") or "").strip()) < 10
        ):
            raise ConceptGraphError(f"schema sync: retired_concepts[{rcid}] missing former_label/former_aliases/replaced_by/reason")
        rep_cid = str(rprov["replaced_by"])
        for fstr in [rprov["former_label"], *rprov["former_aliases"]]:
            fnorm = _norm_prov_tokens(str(fstr))
            if fnorm:
                retired_reserved_norms[fnorm] = rep_cid

    for cid, cnode in concepts_by_id.items():
        c_terms = [str(cnode.get("label") or ""), *(str(a) for a in (cnode.get("aliases") or []))]
        rest_norms_here = restored_alias_norms_by_cid.get(cid, set())
        for tstr in c_terms:
            tnorm = _norm_prov_tokens(tstr)
            if not tnorm:
                continue
            if tnorm in retired_reserved_norms and retired_reserved_norms[tnorm] != cid:
                raise ConceptGraphError(
                    f"schema sync: term {tstr!r} on {cid} collides with permanently reserved retired string (allowed only on {retired_reserved_norms[tnorm]})"
                )
            tomb_owners = global_tombstoned_norms.get(tnorm)
            if tomb_owners and (cid not in tomb_owners or tnorm not in rest_norms_here):
                raise ConceptGraphError(
                    f"schema sync: globally tombstoned alias {tstr!r} appears on {cid} without reviewed restoration"
                )

    for cid, cnode in concepts_by_id.items():
        if cid not in genesis_cids or str(cnode.get("added_in") or "") not in _PRE_SIDECAR_VERSIONS or cid in governed_added_cids:
            cprov = prov_concepts.get(cid)
            if not isinstance(cprov, dict):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} missing term-provenance entry")
            ws = cprov.get("warrant_sources")
            wj = cprov.get("witness_justification")
            if not isinstance(ws, list) or len(ws) < 2 or not isinstance(wj, dict) or set(wj.keys()) != set(_REQUIRED_WITNESS_KEYS):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} missing structured warrant_sources or 6-key witness_justification")
            has_prose = False
            seen_uids: set[str] = set()
            for src in ws:
                if (
                    not isinstance(src, dict)
                    or not str(src.get("unit_id") or "").strip()
                    or src.get("stratum") not in ("S1", "S2", "S3", "S4", "S5")
                    or not str(src.get("provenance_family_id") or "").strip()
                    or src.get("source_type") not in ("prose", "heading")
                    or src.get("origin") not in ("external", "internal")
                    or not str(src.get("locator") or "").strip()
                ):
                    raise ConceptGraphError(f"schema sync: malformed warrant_sources item on {cid}")
                seen_uids.add(str(src["unit_id"]).strip())
                if src.get("source_type") == "prose":
                    has_prose = True
            all_lin, ext_non_ed_lin = _compute_warrant_lineages(ws, burned_doc)
            if len(seen_uids) < 2 or len(all_lin) < 2 or len(ext_non_ed_lin) < 1 or not has_prose:
                raise ConceptGraphError(
                    f"schema sync: post-bootstrap concept {cid} warrant_sources require >=2 distinct units, >=2 independent lineages (<50% author overlap), >=1 non-editor-affiliated external lineage, and >=1 prose source"
                )
            comp_cid = str(wj.get("compared_concept") or "").strip()
            if (
                comp_cid == cid
                or (comp_cid not in concepts_by_id and comp_cid not in retired_by_id)
                or str(wj.get("scenario_a") or "").strip() == str(wj.get("scenario_b") or "").strip()
                or any(len(str(wj.get(k) or "").strip()) < 15 for k in ("criterion_a", "criterion_b", "criterion_c", "scenario_a", "scenario_b"))
            ):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} witness_justification fails minimum completeness checks")

    for cid in sorted(governed_def_updated_cids):
        if cid in concepts_by_id:
            cprov = prov_concepts.get(cid) or {}
            if (
                len(str(cprov.get("edge_impact_note") or "").strip()) < 15
                or cprov.get("definition_change_kind") != "editorial-clarification"
                or not isinstance(cprov.get("checked_incident_relationships"), list)
            ):
                raise ConceptGraphError(
                    f"schema sync: updated definition on {cid} missing edge_impact_note, checked_incident_relationships, or editorial-clarification in term-provenance.yml"
                )

    for cid, expected_added in net_aliases_added.items():
        if cid in concepts_by_id and expected_added:
            cprov = prov_concepts.get(cid) or {}
            ameta_map = cprov.get("alias_metadata") or {}
            active_set = set(concepts_by_id[cid].get("aliases") or [])
            for ak in sorted(expected_added):
                if ak not in active_set or ak not in ameta_map:
                    raise ConceptGraphError(f"schema sync: added or attested alias {ak!r} on {cid} missing from active aliases or term-provenance alias_metadata")

    for cid, expected_removed in all_aliases_removed.items():
        if cid in concepts_by_id and expected_removed:
            cprov = prov_concepts.get(cid) or {}
            rem_map = cprov.get("removed_aliases") or {}
            rest_norms_c = restored_alias_norms_by_cid.get(cid, set())
            active_norms = {_norm_prov_tokens(a) for a in (concepts_by_id[cid].get("aliases") or [])}
            for rk in sorted(expected_removed):
                if rk not in rem_map:
                    raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} missing tombstone in term-provenance removed_aliases")
                rk_norm = _norm_prov_tokens(rk)
                if rk_norm in active_norms and rk_norm not in rest_norms_c:
                    raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} reappeared without reviewed restoration")

    for rid, expected_sec in net_sec_added.items():
        if rid in rows_by_id and expected_sec:
            sprov = prov_subjects.get(rid) or {}
            srat_map = sprov.get("secondary_concepts_rationale") or {}
            active_sec = set(rows_by_id[rid].get("secondary_concepts") or [])
            for scid in sorted(expected_sec):
                if scid not in active_sec or scid not in srat_map:
                    raise ConceptGraphError(f"schema sync: added secondary_concept {scid} on {rid} missing from active row or secondary_concepts_rationale")

    for cid, cprov in prov_concepts.items():
        if cid not in concepts_by_id or not isinstance(cprov, dict) or not cprov:
            raise ConceptGraphError(f"schema sync: invalid or inactive concept {cid} in term-provenance.yml")
        active_aliases = set(concepts_by_id[cid].get("aliases") or [])
        active_alias_norms = {_norm_prov_tokens(a) for a in active_aliases}
        for ak, ameta in (cprov.get("alias_metadata") or {}).items():
            if ak not in active_aliases or not isinstance(ameta, dict):
                raise ConceptGraphError(f"schema sync: term-provenance alias {ak!r} on {cid} missing from active aliases")
            atype = ameta.get("type")
            asrc = str(ameta.get("source") or "").strip()
            snote = str(ameta.get("scope_note") or "").strip()
            cunits = ameta.get("checked_dev_units")
            vq = str(ameta.get("verbatim_quote") or "").strip()
            psha = str(ameta.get("primary_source_sha256") or "").strip()
            ploc = str(ameta.get("primary_source_locator") or "").strip()
            ak_tok = _norm_prov_tokens(ak)
            vq_tok = _norm_prov_tokens(vq)
            if (
                atype not in _ALLOWED_PROV_ALIAS_TYPES
                or len(asrc) < 10
                or len(snote) < 15
                or not isinstance(cunits, list)
                or not cunits
                or any(not isinstance(u, str) or not u.strip() for u in cunits)
                or not _HEX64_RE.match(psha)
                or len(ploc) < 3
                or not ak_tok
                or f" {ak_tok} " not in f" {vq_tok} "
            ):
                raise ConceptGraphError(
                    f"schema sync: incomplete or non-word-bounded alias_metadata on {cid} alias {ak!r}"
                )
            if atype == "upward-posting":
                dlin = ameta.get("dev_lineages_requiring_separation")
                if not isinstance(dlin, int) or isinstance(dlin, bool) or dlin not in (0, 1):
                    raise ConceptGraphError(
                        f"schema sync: upward-posting alias {ak!r} on {cid} requires dev_lineages_requiring_separation in {{0, 1}}"
                    )
        rest_map = cprov.get("restored_aliases") or {}
        if rest_map and not isinstance(rest_map, dict):
            raise ConceptGraphError(f"schema sync: restored_aliases on {cid} must be a mapping")
        rest_map_norms = {_norm_prov_tokens(rk) for rk in rest_map}
        for rk, rrat in (cprov.get("removed_aliases") or {}).items():
            if not isinstance(rrat, str) or len(rrat.strip()) < 15:
                raise ConceptGraphError(f"schema sync: invalid removed_aliases entry {rk!r} on {cid}")
            if _norm_prov_tokens(rk) in active_alias_norms and _norm_prov_tokens(rk) not in rest_map_norms:
                raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} is still active without restored_aliases record")
        if "edge_impact_note" in cprov or "definition_change_kind" in cprov or "checked_incident_relationships" in cprov:
            if (
                len(str(cprov.get("edge_impact_note") or "").strip()) < 15
                or cprov.get("definition_change_kind") != "editorial-clarification"
                or not isinstance(cprov.get("checked_incident_relationships"), list)
                or any(not isinstance(x, str) or not x.strip() for x in cprov["checked_incident_relationships"])
            ):
                raise ConceptGraphError(
                    f"schema sync: concept {cid} definition provenance requires edge_impact_note, checked_incident_relationships, and definition_change_kind='editorial-clarification'"
                )

    for rid, sprov in prov_subjects.items():
        if rid not in rows_by_id or not isinstance(sprov, dict):
            raise ConceptGraphError(f"schema sync: unknown subject {rid} in term-provenance.yml")
        active_sec = set(rows_by_id[rid].get("secondary_concepts") or [])
        for scid, srat in (sprov.get("secondary_concepts_rationale") or {}).items():
            if scid not in active_sec or scid not in concepts_by_id or not isinstance(srat, str) or len(srat.strip()) < 15:
                raise ConceptGraphError(f"schema sync: invalid secondary_concepts_rationale for {scid} on {rid}")

    gf_exc = prov.get("grandfathered_draft_alias_exceptions")
    if gf_exc is not None:
        if not isinstance(gf_exc, dict):
            raise ConceptGraphError("schema sync: grandfathered_draft_alias_exceptions must be a mapping")
        for gcid, grec in gf_exc.items():
            if gcid not in PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS or gcid not in concepts_by_id or not isinstance(grec, dict):
                raise ConceptGraphError(
                    f"schema sync: unpinned or inactive grandfathered_draft_alias_exceptions target {gcid!r} "
                    f"(allowed bootstrap set: {sorted(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS)})"
                )
            galiases = grec.get("aliases")
            active_galiases = set(concepts_by_id[gcid].get("aliases") or [])
            allowed_galiases = set(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS[gcid])
            if (
                not isinstance(galiases, list)
                or not galiases
                or not set(galiases).issubset(allowed_galiases)
                or not set(galiases).issubset(active_galiases)
                or len(str(grec.get("reason") or "").strip()) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: invalid or stale grandfathered_draft_alias_exceptions on {gcid!r}"
                )


def validate_guides_semantics(guides: dict, rows_by_id: dict, version: str | None = None) -> None:
    """Enforce symptom bucket disjointness and (for 0.3-draft+) complete subject role coverage."""
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
        quantities = set(symptom.get("quantities", []))
        overlap_q = sorted(quantities & (causes | checks | levers))
        if overlap_q:
            raise ConceptGraphError(
                f"framework: symptom {sid} lists a quantity also as a cause, check, or lever: {', '.join(overlap_q)}"
            )

    roles = guides.get("roles", [])
    if version not in (None, "0.1-draft", "0.2-draft") and rows_by_id and roles:
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
