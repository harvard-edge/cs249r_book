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
from pathlib import Path
import re
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

PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "mlsys.safety-case": ("assurance case", "UL 4600 safety case"),
    "mlsys.tree-ensemble": ("XGBoost",),
    "mlsys.constrained-decoding": ("grammar-guided decoding",),
    "mlsys.state-sharding": ("ZeRO", "FSDP"),
    "mlsys.paged-kv": ("PagedAttention",),
    "mlsys.graph-replay": ("CUDA Graphs",),
    "mlsys.device-streams": ("CUDA streams",),
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


def canonical_taxonomy_json(doc: dict) -> str:
    """Serialize parsed taxonomy YAML into deterministic key-sorted JSON for semantic equality checks."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_non_guides_taxonomy_sha256(raw_tax: dict) -> str:
    """Compute canonical SHA-256 over all top-level ml-systems-taxonomy.yml sections except guides."""
    non_guides = {k: v for k, v in raw_tax.items() if k != "guides"}
    return hashlib.sha256(canonical_taxonomy_json(non_guides).encode("utf-8")).hexdigest()


def compute_campaign_entry_sha256(entry: dict) -> str:
    """Compute canonical SHA-256 over an applied_campaigns entry excluding campaign_entry_sha256."""
    body = {k: v for k, v in entry.items() if k != "campaign_entry_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
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
    return json.loads(canon_bytes.decode("utf-8"))


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
    return json.loads(Path(burned_path_str).read_text(encoding="utf-8"))


def _load_burned_clusters(root: Path | None = None) -> dict:
    cand = (root / BURNED_CLUSTERS_REL_PATH) if root is not None else None
    path = cand if (cand is not None and cand.exists()) else (REPO_ROOT_PATH / BURNED_CLUSTERS_REL_PATH)
    if not path.exists():
        raise ConceptGraphError(f"schema sync: missing burned-clusters ledger {BURNED_CLUSTERS_REL_PATH}")
    return _cached_burned_clusters(str(path), path.stat().st_mtime_ns)


def _compute_warrant_lineages(warrant_sources: list[dict], burned_doc: dict) -> tuple[set[str], set[str]]:
    """Return `(all_effective_lineages, external_non_editor_effective_lineages)` under `<50%` author overlap."""
    merges = burned_doc.get("cluster_merges") or {}
    internal_set = {merges.get(x, x) for x in (burned_doc.get("internal_clusters") or [])}
    editor_aff_set = {merges.get(x, x) for x in (burned_doc.get("editor_affiliated_clusters") or [])}
    exposed_meta = burned_doc.get("exposed_cluster_metadata") or {}

    author_by_pf: dict[str, set[str]] = {}
    for pf_raw, meta in exposed_meta.items():
        pf = merges.get(pf_raw, pf_raw)
        if pf in internal_set or not isinstance(meta, dict):
            continue
        author_by_pf.setdefault(pf, set()).update(
            " ".join(str(a).strip().lower().split()) for a in (meta.get("authors") or []) if str(a).strip()
        )

    src_pfs: list[tuple[str, bool, bool]] = []
    for src in warrant_sources:
        pf = merges.get(str(src["provenance_family_id"]).strip(), str(src["provenance_family_id"]).strip())
        is_int = (
            pf in internal_set
            or src.get("origin") == "internal"
            or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_"))
        )
        is_ed = pf in editor_aff_set or bool(src.get("editor_affiliated", False))
        if not is_int:
            author_by_pf.setdefault(pf, set()).update(
                " ".join(str(a).strip().lower().split()) for a in (src.get("authors") or []) if str(a).strip()
            )
        src_pfs.append((pf, is_int, is_ed))

    pfs = sorted(author_by_pf.keys())
    parent = {p: p for p in pfs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(pfs)):
        for j in range(i + 1, len(pfs)):
            sa, sb = author_by_pf[pfs[i]], author_by_pf[pfs[j]]
            if sa and sb and (len(sa & sb) / min(len(sa), len(sb))) >= 0.5:
                union(pfs[i], pfs[j])

    all_lineages: set[str] = set()
    ext_non_ed_lineages: set[str] = set()
    for pf, is_int, is_ed in src_pfs:
        if is_int:
            all_lineages.add("pf_internal_mlsysbook")
        else:
            eff = f"eff:{find(pf)}"
            all_lineages.add(eff)
            if not is_ed:
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
            expected_ops.append("version")
            doc["version"] = str(change["version"]).strip()

        return {"expected_ops": sorted(expected_ops)}

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

    if change.get("version"):
        expected_ops.append("version")
        doc["version"] = str(change["version"]).strip()

    return {
        "expected_ops": sorted(expected_ops),
        "concepts_added": sorted(exp_c_added),
        "concepts_retired": sorted(exp_c_retired),
        "definitions_updated": sorted(exp_defs_updated),
        "aliases_added": {k: list(v) for k, v in sorted(exp_aliases_added.items())},
        "aliases_removed": {k: list(v) for k, v in sorted(exp_aliases_removed.items())},
        "other_concept_updates": sorted(exp_other_c_upd),
        "secondary_concepts_added": {k: list(v) for k, v in sorted(exp_sec_added.items())},
        "secondary_concepts_removed": {k: list(v) for k, v in sorted(exp_sec_removed.items())},
        "other_subject_updates": sorted(exp_other_s_upd),
    }


def validate_term_provenance_sidecar(root: Path) -> None:
    """Verify hash-chained provenance, deterministic genesis replay, and Gate 2/5 records in term-provenance.yml.

    Security & integrity boundary: `sidecar_content_sha256`, `campaign_entry_sha256`, and bound script/file
    digests act as in-repo consistency and deterministic replay-integrity checks (detecting un-reconciled
    YAML edits, partial writes, or untracked edits to 0.4-draft concepts). External authorization against a
    committer modifying validator code is supplied by git history and CI diff review against `origin/dev`.
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

    body_without_self = {k: v for k, v in prov.items() if k != "sidecar_content_sha256"}
    expected_sidecar_sha = hashlib.sha256(
        json.dumps(body_without_self, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if prov.get("sidecar_content_sha256") != expected_sidecar_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/term-provenance.yml sidecar_content_sha256 mismatch (hand-edited outside apply.py)"
        )

    campaigns = prov.get("applied_campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ConceptGraphError("schema sync: taxonomy/term-provenance.yml must record non-empty applied_campaigns hash chain")

    # Verify top-level post_apply_taxonomy_sha256 / post_non_guides_taxonomy_sha256 chain endpoints first.
    if isinstance(campaigns[-1], dict):
        if campaigns[-1].get("post_apply_taxonomy_sha256") != expected_tax_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_apply_taxonomy_sha256 disagrees with taxonomy_sha256")
        if campaigns[-1].get("post_non_guides_taxonomy_sha256") != expected_non_guides_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_non_guides_taxonomy_sha256 disagrees with non_guides_taxonomy_sha256")

    genesis_doc = load_genesis_baseline_taxonomy(root)
    genesis_cids = {c["id"] for c in genesis_doc.get("concepts", [])}
    burned_doc = _load_burned_clusters(root)
    replayed_doc = deepcopy(genesis_doc)

    governed_added_cids: set[str] = set()
    governed_touched_cids: set[str] = set()
    governed_retired_cids: set[str] = set()
    governed_def_updated_cids: set[str] = set()
    net_aliases_added: dict[str, set[str]] = {}
    all_aliases_removed: dict[str, set[str]] = {}
    restored_aliases_by_cid: dict[str, set[str]] = {}
    global_tombstoned_norms: dict[str, set[str]] = {}
    net_sec_added: dict[str, set[str]] = {}
    last_full_rec: dict | None = None
    unclosed_pre_eval = False

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
            if prev_entry_sha != campaigns[idx - 1].get("campaign_entry_sha256"):
                raise ConceptGraphError(f"schema sync: broken previous_campaign_entry_sha256 chain at applied_campaigns[{idx}]")

        replayed_patch = entry.get("replayed_patch")
        if not isinstance(replayed_patch, dict) or not replayed_patch:
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing non-empty replayed_patch")

        if entry.get("allow_pre_evaluation_apply") is True:
            if replayed_patch.get("version"):
                raise ConceptGraphError(
                    f"schema sync: allow_pre_evaluation_apply is forbidden with a version bump at applied_campaigns[{idx}]"
                )
            unclosed_pre_eval = True
        if entry.get("evaluation_gate_completed") is True:
            unclosed_pre_eval = False

        if mode == "full-3family-reconcile":
            if entry.get("receipt_schema_version") != 4:
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must have receipt_schema_version=4")
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
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing valid {hkey}")
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

            c_added = sorted(details.get("concepts_added") or [])
            c_retired = sorted(details.get("concepts_retired") or [])
            d_upd = sorted(details.get("definitions_updated") or [])
            a_added_map = {k: list(v) for k, v in sorted((details.get("aliases_added") or {}).items())}
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
                or a_rem_map != replay_meta["aliases_removed"]
                or other_c_upd != replay_meta["other_concept_updates"]
                or s_add_map != replay_meta["secondary_concepts_added"]
                or s_rem_map != replay_meta["secondary_concepts_removed"]
                or other_s_upd != replay_meta["other_subject_updates"]
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
            for cid in set(d_upd) | set(a_added_map) | set(a_rem_map):
                governed_touched_cids.add(cid)
            for rcid in c_retired:
                governed_retired_cids.add(rcid)
                governed_added_cids.discard(rcid)
                governed_def_updated_cids.discard(rcid)
                net_aliases_added.pop(rcid, None)

            governed_def_updated_cids.update(d_upd)
            for cid, alist in a_added_map.items():
                net_aliases_added.setdefault(cid, set()).update(alist)
            for cid, rlist in a_rest_map.items():
                restored_aliases_by_cid.setdefault(cid, set()).update(rlist)
                for ra in rlist:
                    rn = _norm_prov_tokens(ra)
                    if rn in global_tombstoned_norms:
                        global_tombstoned_norms[rn].discard(cid)
            for cid, rlist in a_rem_map.items():
                all_aliases_removed.setdefault(cid, set()).update(rlist)
                for ra in rlist:
                    global_tombstoned_norms.setdefault(_norm_prov_tokens(ra), set()).add(cid)
                if cid in net_aliases_added:
                    net_aliases_added[cid].difference_update(rlist)
                if cid in restored_aliases_by_cid:
                    restored_aliases_by_cid[cid].difference_update(rlist)
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
                "guides_reconcile_receipt_sha256",
                "guides_apply_script_sha256",
                "workspace_isolation_guides_sha256",
                "run_reviews_g1_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply missing valid {hkey}")
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
                if gop != "version" and not gop.startswith(_ALLOWED_GUIDES_OP_PREFIXES):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] unrecognized guides operation {gop!r}")

            step_base_ver = str(replayed_doc.get("version") or "")
            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if sorted(gops) != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides_operations_applied disagrees with replayed_patch"
                )
            if "version" in gops:
                ng_with_base_ver = {**replayed_doc, "version": step_base_ver}
                if compute_non_guides_taxonomy_sha256(ng_with_base_ver) != base_ng_sha:
                    raise ConceptGraphError(
                        f"schema sync: applied_campaigns[{idx}] guides-apply modified non-guides content beyond version"
                    )
            elif post_ng_sha != base_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides-apply modified non-guides taxonomy content"
                )
            if compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] replayed guides non_guides_taxonomy_sha256 mismatch"
                )
        else:
            raise ConceptGraphError(f"schema sync: unsupported applied_campaigns mode {mode!r}")

    if last_full_rec is None or prov.get("last_reconcile_receipt") != last_full_rec:
        raise ConceptGraphError("schema sync: last_reconcile_receipt disagrees with latest full-3family-reconcile campaign entry")

    if canonical_taxonomy_json(replayed_doc) != canonical_taxonomy_json(raw_tax):
        raise ConceptGraphError(
            "schema sync: replaying applied_campaigns from genesis 0.4-draft does not reproduce ml-systems-taxonomy.yml "
            "(untracked edit outside governed patches)"
        )

    is_non_draft = not str(raw_tax.get("version") or "").endswith("-draft")
    if unclosed_pre_eval and (is_non_draft or prov.get("release_candidate_ready") is True):
        raise ConceptGraphError(
            "schema sync: unclosed allow_pre_evaluation_apply override cannot ship in a release candidate or final release"
        )
    if bool(prov.get("pending_pre_evaluation_override", False)) != unclosed_pre_eval:
        raise ConceptGraphError(
            "schema sync: pending_pre_evaluation_override flag disagrees with applied_campaigns pre-evaluation state"
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

    # Permanent retired-string reservation across all retired concepts (even after successor adopts/removes).
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

    # Global removed-alias tombstone enforcement + retired-string reservation across all active concepts.
    for cid, cnode in concepts_by_id.items():
        c_terms = [str(cnode.get("label") or ""), *(str(a) for a in (cnode.get("aliases") or []))]
        rest_norms_here = {_norm_prov_tokens(x) for x in restored_aliases_by_cid.get(cid, set())}
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
            ):
                raise ConceptGraphError(f"schema sync: updated definition on {cid} missing edge_impact_note or editorial-clarification in term-provenance.yml")

    for cid, expected_added in net_aliases_added.items():
        if cid in concepts_by_id and expected_added:
            cprov = prov_concepts.get(cid) or {}
            ameta_map = cprov.get("alias_metadata") or {}
            active_set = set(concepts_by_id[cid].get("aliases") or [])
            for ak in sorted(expected_added):
                if ak not in active_set or ak not in ameta_map:
                    raise ConceptGraphError(f"schema sync: added alias {ak!r} on {cid} missing from active aliases or term-provenance alias_metadata")

    for cid, expected_removed in all_aliases_removed.items():
        if cid in concepts_by_id and expected_removed:
            cprov = prov_concepts.get(cid) or {}
            rem_map = cprov.get("removed_aliases") or {}
            rest_set = restored_aliases_by_cid.get(cid, set())
            active_set = set(concepts_by_id[cid].get("aliases") or [])
            for rk in sorted(expected_removed):
                if rk not in rem_map:
                    raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} missing tombstone in term-provenance removed_aliases")
                if rk in active_set and rk not in rest_set:
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
        for rk, rrat in (cprov.get("removed_aliases") or {}).items():
            if not isinstance(rrat, str) or len(rrat.strip()) < 15:
                raise ConceptGraphError(f"schema sync: invalid removed_aliases entry {rk!r} on {cid}")
            if rk in active_aliases and rk not in rest_map:
                raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} is still active without restored_aliases record")
        if "edge_impact_note" in cprov or "definition_change_kind" in cprov:
            if (
                len(str(cprov.get("edge_impact_note") or "").strip()) < 15
                or cprov.get("definition_change_kind") != "editorial-clarification"
            ):
                raise ConceptGraphError(
                    f"schema sync: concept {cid} definition provenance requires edge_impact_note and definition_change_kind='editorial-clarification'"
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
