"""LinkML compilation, current structural checks, and graph invariants."""
from __future__ import annotations

import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import jsonschema

from ..concept_graph import ConceptGraphError
from .provenance import (
    validate_term_provenance_sidecar,
)
from .records import (
    _SEMVER_RE,
    parse_taxonomy_semver,
)

try:
    from linkml_runtime.utils.schemaview import SchemaView
    HAS_LINKML = True
except ModuleNotFoundError:  # pragma: no cover
    SchemaView = None
    HAS_LINKML = False


SCHEMA_REL_PATH = Path("taxonomy/schema/mlsys-taxonomy.linkml.yml")


REPO_SCHEMA_PATH = Path(__file__).resolve().parents[4] / SCHEMA_REL_PATH


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
    from ..field_graph import load_taxonomy_yaml

    loaded = load_taxonomy_yaml(Path(schema_path_str))
    if not isinstance(loaded, dict):
        raise ConceptGraphError(f"{schema_path_str}: schema must be a YAML mapping")
    return loaded


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
    from ..taxonomy_coverage import STATUSES

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
                "schema sync: LinkML RelationIdEnum disagrees with taxonomy.yml relations"
            )
        for rel in relations:
            rid = rel["id"]
            pv = rel_pvs.get(rid) or {}
            ann = pv.get("annotations") or {}
            if rel.get("meaning") != pv.get("description"):
                raise ConceptGraphError(
                    f"schema sync: RelationIdEnum.{rid} description disagrees with taxonomy.yml meaning"
                )
            for field_key in ("kind", "forward", "inverse"):
                if rel.get(field_key) != ann.get(field_key):
                    raise ConceptGraphError(
                        f"schema sync: RelationIdEnum.{rid} annotation '{field_key}' disagrees with taxonomy.yml"
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
    prov_path = root / "taxonomy/provenance.yml"
    if prov_path.exists() or readme_path.exists():
        validate_term_provenance_sidecar(root)


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
