"""Read the ML Systems Taxonomy that question tags refer to.

The canonical single source of truth lives at ``taxonomy/ml-systems-taxonomy.yml``
(governed by ``taxonomy/schema/mlsys-taxonomy.linkml.yml``), two levels above
``staffml/vault``. When ``binder/cli`` is present in the repository checkout,
this module delegates directly to ``binder.cli.core.field_graph.load_unified_taxonomy``
and ``binder.cli.core.taxonomy_release.staffml_labels_payload`` so StaffML and Binder
share a single loader, validator, and exporter implementation.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_CONCEPT_RE = re.compile(r"^mlsys\.[a-z0-9]+(-[a-z0-9]+)*$")
_SCOPE_REF_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_INCREASING_VERBS = re.compile(
    r"\b(increase|increases|increasing|raise|raises|raising|grow|grows|growing|inflate|inflates|inflating|amplify|amplifies|amplifying|worsen|worsens|worsening|exacerbate|exacerbates|exacerbating|multiply|multiplies|multiplying)\b",
    re.IGNORECASE,
)
_DECREASING_VERBS = re.compile(
    r"\b(shrink|shrinks|shrinking|lower|lowers|lowering|reduce|reduces|reducing|decrease|decreases|decreasing|curb|curbs|curbing|cut|cuts|cutting|eliminate|eliminates|eliminating|minimize|minimizes|minimizing|compress|compresses|compressing|invalidate|invalidates|invalidating)\b",
    re.IGNORECASE,
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML SafeLoader that rejects duplicate keys in any mapping."""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class Taxonomy:
    """Validated projection of the ML Systems Taxonomy (`framework` and `field` aspects)."""

    framework: dict[str, Any]
    field: dict[str, Any]

    @property
    def subjects(self) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Subject id → (subject row, its area)."""
        return {row["id"]: (row, area) for area in self.framework.get("areas", []) for row in area.get("rows", [])}

    @property
    def concepts(self) -> dict[str, dict[str, Any]]:
        """Concept id → concept record."""
        return {c["id"]: c for c in self.field.get("concepts", [])}

    @property
    def retired(self) -> dict[str, str]:
        """Retired concept id → the concept that replaces it."""
        return {r["id"]: r.get("replaced_by", "") for r in self.field.get("retired", []) or []}

    @property
    def home(self) -> dict[str, str]:
        """Concept id → the subject id that is its home."""
        return {c: rid for rid, (row, _) in self.subjects.items() for c in row.get("concepts", [])}


def _import_binder_core(repo_root: Path):
    binder_cli = repo_root / "binder" / "cli"
    if not binder_cli.is_dir():
        return None
    if str(binder_cli) not in sys.path:
        sys.path.insert(0, str(binder_cli))
    try:
        from core.concept_graph import ConceptGraphError  # type: ignore[import-not-found]
        from core.field_graph import load_unified_taxonomy  # type: ignore[import-not-found]
        from core.taxonomy_release import staffml_labels_payload  # type: ignore[import-not-found]

        return load_unified_taxonomy, staffml_labels_payload, ConceptGraphError
    except ImportError:
        return None


def _validate_fallback_linkml(raw: dict[str, Any], schema_doc: dict[str, Any], unified_path: Path) -> None:
    """Validate raw taxonomy against UnifiedTaxonomyModel in schema_doc using jsonschema."""
    import jsonschema

    classes = schema_doc.get("classes") or {}
    enums = schema_doc.get("enums") or {}
    defs: dict[str, Any] = {}
    for ename, edef in enums.items():
        pvs = list(((edef or {}).get("permissible_values") or {}).keys())
        defs[ename] = {"type": "string", "enum": pvs}
    for cname, cdef in classes.items():
        if cname == "AnyValue":
            defs[cname] = {"type": "object"}
            continue
        props: dict[str, Any] = {}
        req: list[str] = []
        for sname, sdef in ((cdef or {}).get("attributes") or {}).items():
            sdef = sdef or {}
            rng = sdef.get("range", "string")
            inlined = bool(sdef.get("inlined") or sdef.get("inlined_as_list"))
            ann = sdef.get("annotations") or {}
            if ann:
                item: dict[str, Any] = {"type": "object"}
                if "dict_key_pattern" in ann:
                    item["propertyNames"] = {"pattern": ann["dict_key_pattern"]}
                if "dict_value_ref" in ann:
                    item["additionalProperties"] = {"$ref": f"#/$defs/{ann['dict_value_ref']}"}
                elif "dict_value_array_ref" in ann:
                    item["additionalProperties"] = {
                        "type": "array",
                        "items": {"$ref": f"#/$defs/{ann['dict_value_array_ref']}"},
                    }
            elif rng == "integer":
                item = {"type": "integer"}
                if "equals_number" in sdef:
                    item["const"] = sdef["equals_number"]
                if "minimum_value" in sdef:
                    item["minimum"] = sdef["minimum_value"]
                if "maximum_value" in sdef:
                    item["maximum"] = sdef["maximum_value"]
            elif rng == "boolean":
                item = {"type": "boolean"}
            elif rng in enums:
                if "equals_string" in sdef or "pattern" in sdef:
                    sub: dict[str, Any] = {}
                    if "equals_string" in sdef:
                        sub["const"] = sdef["equals_string"]
                    if "pattern" in sdef:
                        sub["pattern"] = sdef["pattern"]
                    item = {"allOf": [{"$ref": f"#/$defs/{rng}"}, sub]}
                else:
                    item = {"$ref": f"#/$defs/{rng}"}
            elif rng in classes and inlined:
                item = {"$ref": f"#/$defs/{rng}"}
            else:
                item = {"type": "string"}
                if rng in classes:
                    target_attrs = ((classes[rng] or {}).get("attributes")) or {}
                    for tsdef in target_attrs.values():
                        if isinstance(tsdef, dict) and tsdef.get("identifier") and isinstance(tsdef.get("pattern"), str):
                            item["pattern"] = tsdef["pattern"]
                            break
                if "equals_string" in sdef:
                    item["const"] = sdef["equals_string"]
                if "pattern" in sdef:
                    item["pattern"] = sdef["pattern"]
            if sdef.get("multivalued"):
                prop: dict[str, Any] = (
                    {"type": "array", "items": item}
                    if sdef.get("required")
                    else {"type": ["array", "null"], "items": item}
                )
                if "minimum_cardinality" in sdef:
                    prop["minItems"] = sdef["minimum_cardinality"]
                if "maximum_cardinality" in sdef:
                    prop["maxItems"] = sdef["maximum_cardinality"]
            else:
                prop = (
                    dict(item)
                    if sdef.get("required")
                    else {"anyOf": [item, {"type": "null"}]}
                )
            props[sname] = prop
            if sdef.get("required"):
                req.append(sname)
        cls_schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
        if req:
            cls_schema["required"] = req
        defs[cname] = cls_schema
    if "UnifiedTaxonomyModel" in defs:
        root_schema = {"$schema": "https://json-schema.org/draft/2019-09/schema", "$defs": defs, **defs["UnifiedTaxonomyModel"]}
        errors = list(jsonschema.Draft201909Validator(root_schema).iter_errors(raw))
        if errors:
            raise ValueError(f"{unified_path}: LinkML schema validation failed: {errors[0].message}")


def _find_cycle(adj: dict[str, list[str]]) -> str | None:
    """Return a node participating in a directed cycle in `adj`, or `None` if `adj` is a DAG."""
    state: dict[str, int] = {}
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
                if state.get(nxt, 0) == 1:
                    return nxt
                if state.get(nxt, 0) == 0:
                    state[nxt] = 1
                    stack.append((nxt, 0))
            else:
                state[node] = 2
                stack.pop()
    return None


def _load_standalone_taxonomy(repo_root: Path, unified_path: Path) -> Taxonomy:
    """Validate and project `taxonomy/ml-systems-taxonomy.yml` when `binder/cli` is not available."""
    text = unified_path.read_text(encoding="utf-8")
    if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text, Loader=yaml.SafeLoader)):
        raise ValueError(f"{unified_path.name}: YAML anchors/aliases are not allowed")
    raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict) or "areas" not in raw or "concepts" not in raw:
        raise ValueError(f"{unified_path}: malformed taxonomy file (requires 'areas' and 'concepts')")
    if raw.get("schema_version", 1) != 1:
        raise ValueError(f"{unified_path}: unsupported schema_version {raw.get('schema_version')}")

    schema_path = repo_root / "taxonomy" / "schema" / "mlsys-taxonomy.linkml.yml"
    schema_doc: dict[str, Any] = (
        yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
        if schema_path.exists()
        else {}
    )
    rel_pvs = ((schema_doc.get("enums") or {}).get("RelationIdEnum") or {}).get("permissible_values") or {}
    hierarchical_rels = tuple(
        rel_id
        for rel_id, pv in rel_pvs.items()
        if isinstance(pv, dict) and isinstance(pv.get("annotations"), dict) and pv["annotations"].get("hierarchical_dag") is True
    ) or ("specializes", "part-of", "requires")

    known_rows: set[str] = set()
    for area in raw.get("areas", []):
        if isinstance(area, dict):
            for row in area.get("rows", []):
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    rid = row["id"]
                    if not _SLUG_RE.fullmatch(rid):
                        raise ValueError(f"{unified_path}: invalid subject row id {rid!r}")
                    if rid in known_rows:
                        raise ValueError(f"{unified_path}: duplicate subject row id {rid!r}")
                    known_rows.add(rid)
    families = {
        f["id"]
        for f in raw.get("families", []) or []
        if isinstance(f, dict) and isinstance(f.get("id"), str)
    }
    by_home: dict[str, list[str]] = {}
    concept_home: dict[str, str] = {}
    field_concepts: list[dict[str, Any]] = []
    named_terms: dict[str, str] = {}
    for c in raw.get("concepts", []):
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not _CONCEPT_RE.fullmatch(c["id"]):
            raise ValueError(f"{unified_path}: invalid concept record {c!r}")
        cid = c["id"]
        if cid in concept_home:
            raise ValueError(f"{unified_path}: duplicate concept id {cid!r}")
        if families and "family" in c and c.get("family") not in families:
            raise ValueError(f"{unified_path}: concept {cid} has unknown family {c.get('family')!r}")
        home = c.get("home")
        if not isinstance(home, str) or home not in known_rows:
            raise ValueError(f"{unified_path}: concept {cid} has missing or unknown home {home!r}")
        for raw_term in [c.get("label", ""), *(c.get("aliases") or [])]:
            if isinstance(raw_term, str) and raw_term.strip():
                norm_term = " ".join(raw_term.lower().split())
                if norm_term in named_terms:
                    raise ValueError(f"{unified_path}: '{norm_term}' names both {named_terms[norm_term]} and {cid}")
                named_terms[norm_term] = cid
        concept_home[cid] = home
        by_home.setdefault(home, []).append(cid)
        field_concepts.append({k: v for k, v in c.items() if k != "home"})
    for ex in raw.get("placement_examples", []) or []:
        if not isinstance(ex, dict) or ex.get("concept") not in concept_home or concept_home[ex["concept"]] != ex.get("home"):
            raise ValueError(f"{unified_path}: placement_examples mismatch on {ex!r}")
    edges = raw.get("relationships") or []
    rel_by_id = {
        r["id"]: r
        for r in (raw.get("relations") or [])
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }
    seen_rel_ids: set[str] = set()
    seen_rel_pairs: set[frozenset[str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("id"), str):
            raise ValueError(f"{unified_path}: invalid relationship {edge!r}")
        eid = edge["id"]
        if eid in seen_rel_ids:
            raise ValueError(f"{unified_path}: duplicate relationship id {eid!r}")
        seen_rel_ids.add(eid)
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in concept_home or tgt not in concept_home or src == tgt:
            raise ValueError(f"{unified_path}: relationship {eid} has missing or identical endpoints")
        pair = frozenset((src, tgt))
        if pair in seen_rel_pairs:
            raise ValueError(f"{unified_path}: duplicate concept pair on relationship {eid!r}")
        seen_rel_pairs.add(pair)
        rel_verb = edge.get("relation")
        if rel_by_id and (rel_verb not in rel_by_id or edge.get("kind") != rel_by_id[rel_verb].get("kind")):
            raise ValueError(f"{unified_path}: relationship {eid} has invalid relation or kind")
        predicate = str(edge.get("predicate", ""))
        if rel_verb == "increases" and _DECREASING_VERBS.search(predicate) and not _INCREASING_VERBS.search(predicate):
            raise ValueError(f"{unified_path}: relationship {eid} predicate contradicts 'increases'")
        if rel_verb in ("reduces", "mitigates", "limits") and _INCREASING_VERBS.search(predicate) and not _DECREASING_VERBS.search(predicate):
            raise ValueError(f"{unified_path}: relationship {eid} predicate contradicts {rel_verb!r}")
    for rec in raw.get("retired", []) or []:
        if isinstance(rec, dict) and (rec.get("id") in concept_home or rec.get("replaced_by") not in concept_home):
            raise ValueError(f"{unified_path}: retired identity {rec.get('id')} must be unused and name a current replacement")
    if raw.get("version") not in (None, "0.1-draft", "0.2-draft") and concept_home:
        deg: dict[str, int] = {cid: 0 for cid in concept_home}
        undir: dict[str, set[str]] = {cid: set() for cid in concept_home}
        for edge in edges:
            s_id, t_id = edge["source"], edge["target"]
            deg[s_id] = deg.get(s_id, 0) + 1
            deg[t_id] = deg.get(t_id, 0) + 1
            undir[s_id].add(t_id)
            undir[t_id].add(s_id)
        low_deg = sorted(cid for cid, d in deg.items() if d < 2)
        if low_deg:
            raise ValueError(f"{unified_path}: concepts need at least two relationships: {', '.join(low_deg[:5])}")
        start_c = next(iter(concept_home))
        visited_c = {start_c}
        queue_c = [start_c]
        while queue_c:
            cur_c = queue_c.pop()
            for nxt_c in undir.get(cur_c, ()):
                if nxt_c not in visited_c:
                    visited_c.add(nxt_c)
                    queue_c.append(nxt_c)
        if len(visited_c) != len(concept_home):
            raise ValueError(f"{unified_path}: concept relationship graph is disconnected")
    for rel_name in hierarchical_rels:
        adj: dict[str, list[str]] = {}
        for edge in edges:
            if edge.get("relation") == rel_name:
                adj.setdefault(edge["source"], []).append(edge["target"])
        if _find_cycle(adj) is not None:
            raise ValueError(f"{unified_path}: ontological cycle in {rel_name!r}")
    guides_raw = raw.get("guides") or {}
    if isinstance(guides_raw, dict):
        neighbors: dict[str, set[str]] = {cid: set() for cid in concept_home}
        for edge in edges:
            neighbors.setdefault(edge["source"], set()).add(edge["target"])
            neighbors.setdefault(edge["target"], set()).add(edge["source"])
        for role in guides_raw.get("roles", []) or []:
            if isinstance(role, dict):
                core_s = role.get("core", []) or []
                work_s = role.get("working", []) or []
                if any(s not in known_rows for s in [*core_s, *work_s]) or (set(core_s) & set(work_s)):
                    raise ValueError(f"{unified_path}: invalid role guide {role.get('id')}")
        for symptom in guides_raw.get("symptoms", []) or []:
            if isinstance(symptom, dict):
                causes = set(symptom.get("causes", []) or [])
                checks = set(symptom.get("checks", []) or [])
                levers = set(symptom.get("levers", []) or [])
                quantities = set(symptom.get("quantities", []) or [])  # the measured metric, kept apart from causes (2026-11-02)
                if any(c not in concept_home for c in (causes | checks | levers | quantities)):
                    raise ValueError(f"{unified_path}: symptom {symptom.get('id')} references unknown concept")
                if (causes & levers) or (causes & checks) or (checks & levers) or (quantities & (causes | checks | levers)):
                    raise ValueError(f"{unified_path}: symptom {symptom.get('id')} has overlapping cause/check/lever/quantity buckets")
                for item in checks | levers:
                    if not (neighbors.get(item, set()) & (causes | quantities)):
                        raise ValueError(f"{unified_path}: symptom {symptom.get('id')} check/lever {item} shares no relationship with a cause or quantity")
        for path_g in guides_raw.get("learning_paths", []) or []:
            if isinstance(path_g, dict):
                steps = [st.get("subject") for st in (path_g.get("steps") or []) if isinstance(st, dict)]
                if any(s not in known_rows for s in steps) or len(set(steps)) != len(steps):
                    raise ValueError(f"{unified_path}: invalid learning path guide {path_g.get('id')}")
        roles_list = guides_raw.get("roles", []) or []
        if raw.get("version") not in (None, "0.1-draft", "0.2-draft") and known_rows and roles_list:
            covered_rows = {
                subj
                for r in roles_list if isinstance(r, dict)
                for subj in (*(r.get("core", []) or []), *(r.get("working", []) or []))
            }
            missing_rows = sorted(known_rows - covered_rows)
            if missing_rows:
                raise ValueError(f"{unified_path}: subjects missing from all role guides: {', '.join(missing_rows)}")
    known_areas = {a["id"] for a in (raw.get("areas") or []) if isinstance(a, dict) and isinstance(a.get("id"), str)}
    known_area_and_row_ids = known_areas | known_rows
    for area in raw.get("areas", []):
        if isinstance(area, dict):
            for row in area.get("rows", []):
                if isinstance(row, dict):
                    for sec_c in row.get("secondary_concepts", []) or []:
                        if sec_c not in concept_home or concept_home.get(sec_c) == row.get("id"):
                            raise ValueError(f"{unified_path}: invalid secondary_concepts {sec_c!r} on {row.get('id')}")
                    for scope_line in [*(row.get("includes") or []), *(row.get("excludes") or [])]:
                        if isinstance(scope_line, str):
                            for ref_id in _SCOPE_REF_RE.findall(scope_line):
                                if ref_id not in known_area_and_row_ids:
                                    raise ValueError(f"{unified_path}: row {row.get('id')} cites unknown row or area {ref_id}")
    if any(isinstance(a, dict) and "code" in a for a in raw.get("areas", [])):
        groups_by_id = {g["id"]: g for g in (raw.get("groups") or []) if isinstance(g, dict) and "id" in g}
        seen_codes: set[str] = set()
        for g in raw.get("groups") or []:
            if isinstance(g, dict):
                c_val = g.get("code")
                if not isinstance(c_val, str) or not re.fullmatch(r"[A-Z]", c_val) or c_val in seen_codes:
                    raise ValueError(f"{unified_path}: invalid or duplicate group code on {g.get('id')}")
                seen_codes.add(c_val)
        for area in raw.get("areas", []):
            if isinstance(area, dict):
                a_code = area.get("code")
                if not isinstance(a_code, str) or not re.fullmatch(r"[A-Z]\d+", a_code) or a_code in seen_codes:
                    raise ValueError(f"{unified_path}: invalid or duplicate area code on {area.get('id')}")
                seen_codes.add(a_code)
                grp = groups_by_id.get(area.get("group", ""))
                if grp and not a_code.startswith(str(grp.get("code", ""))):
                    raise ValueError(f"{unified_path}: area code {a_code} does not extend group code")
                for row in area.get("rows", []):
                    if isinstance(row, dict):
                        r_code = row.get("code")
                        if not isinstance(r_code, str) or not re.fullmatch(r"[A-Z]\d+\.\d+", r_code) or r_code in seen_codes:
                            raise ValueError(f"{unified_path}: invalid or duplicate subject code on {row.get('id')}")
                        seen_codes.add(r_code)
                        if not r_code.startswith(f"{a_code}."):
                            raise ValueError(f"{unified_path}: subject code {r_code} does not extend area code")
    skos_raw = raw.get("skos_hierarchy")
    if isinstance(skos_raw, dict):
        topics_by_id: dict[str, dict[str, Any]] = {}
        for t in skos_raw.get("topics", []) or []:
            if isinstance(t, dict) and isinstance(t.get("id"), str):
                tid = t["id"]
                if tid in topics_by_id or tid in concept_home:
                    raise ValueError(f"{unified_path}: duplicate or overlapping SKOS topic {tid}")
                topics_by_id[tid] = t
        b_adj: dict[str, list[str]] = {}
        seen_hier: set[tuple[str, str]] = set()
        for pair in skos_raw.get("hierarchy", []) or []:
            if isinstance(pair, dict):
                n_id, b_id = pair.get("narrower"), pair.get("broader")
                if (
                    not isinstance(n_id, str)
                    or not isinstance(b_id, str)
                    or (n_id not in concept_home and n_id not in topics_by_id)
                    or b_id not in topics_by_id
                    or n_id == b_id
                    or (n_id, b_id) in seen_hier
                ):
                    raise ValueError(f"{unified_path}: invalid skos_hierarchy assertion {pair!r}")
                seen_hier.add((n_id, b_id))
                b_adj.setdefault(b_id, []).append(n_id)
        cyc_node = _find_cycle(b_adj)
        if cyc_node is not None:
            raise ValueError(f"{unified_path}: cycle in skos_hierarchy at {cyc_node}")
        col_families = [
            col.get("family")
            for col in (skos_raw.get("collections") or [])
            if isinstance(col, dict)
        ]
        if families and (set(col_families) != families or len(col_families) != len(families)):
            raise ValueError(f"{unified_path}: skos_hierarchy collections must cover every family once")
    if schema_doc:
        _validate_fallback_linkml(raw, schema_doc, unified_path)
        if str(raw.get("version", "")) != str(schema_doc.get("version", "")):
            raise ValueError(f"{unified_path}: taxonomy version disagrees with LinkML schema version")
        rels = raw.get("relations") or []
        if {r.get("id") for r in rels if isinstance(r, dict)} != set(rel_pvs.keys()) or len(rels) != len(rel_pvs):
            raise ValueError(f"{unified_path}: relations disagree with RelationIdEnum")
        for r in rels:
            pv = rel_pvs.get(r["id"]) or {}
            ann = pv.get("annotations") or {}
            if r.get("meaning") != pv.get("description") or any(r.get(k) != ann.get(k) for k in ("kind", "forward", "inverse")):
                raise ValueError(f"{unified_path}: relation {r['id']} metadata disagrees with RelationIdEnum")
    resolved_areas = []
    for area in raw.get("areas", []):
        rows = []
        for row in area.get("rows", []):
            if "concepts" in row:
                raise ValueError(f"{unified_path}: subject row {row.get('id')} must not hand-list concepts")
            rows.append({**row, "concepts": list(by_home.get(row.get("id", ""), []))})
        resolved_areas.append({**area, "rows": rows})
    framework = {**raw, "areas": resolved_areas}
    field = {
        "concepts": field_concepts,
        "retired": raw.get("retired", []),
    }
    return Taxonomy(framework, field)


def load_taxonomy(vault_dir: Path) -> Taxonomy | None:
    """Return the repo's validated taxonomy, or None in a vault-only checkout."""
    repo_root = vault_dir.parent.parent
    unified_path = repo_root / "taxonomy" / "ml-systems-taxonomy.yml"
    if not unified_path.exists():
        return None

    binder_core = _import_binder_core(repo_root)
    if binder_core is not None:
        load_unified_taxonomy, _, concept_graph_error_cls = binder_core
        try:
            bundle = load_unified_taxonomy(repo_root)
        except concept_graph_error_cls as exc:
            raise ValueError(str(exc)) from exc
        if bundle is None or not bundle.get("framework") or not bundle.get("field"):
            raise ValueError(f"{unified_path}: malformed taxonomy file (requires 'areas' and 'concepts')")
        return Taxonomy(bundle["framework"], bundle["field"])

    return _load_standalone_taxonomy(repo_root, unified_path)


def build_staffml_labels_payload(vault_dir: Path, taxonomy: Taxonomy) -> dict[str, Any]:
    """Build the StaffML taxonomy labels payload using the canonical Binder release builder."""
    repo_root = vault_dir.parent.parent
    binder_core = _import_binder_core(repo_root)
    if binder_core is not None:
        _, staffml_labels_payload, _ = binder_core
        return staffml_labels_payload(taxonomy.framework, taxonomy.field)
    home = taxonomy.home
    subjects = {
        rid: {
            "code": row.get("code", ""),
            "label": row["label"],
            "short": row.get("display_label") or row["label"],
            "area": area["label"],
            "area_id": area.get("code", ""),
            "area_slug": area["id"],
            "definition": row.get("definition", ""),
        }
        for rid, (row, area) in taxonomy.subjects.items()
    }
    concepts = {cid: {"label": c["label"], "subject": home.get(cid, "")} for cid, c in taxonomy.concepts.items()}
    groups = {g["id"]: g for g in taxonomy.framework.get("groups", [])}
    areas = [
        {
            "id": a.get("code", ""),
            "slug": a["id"],
            "label": a["label"],
            "short": a.get("display_label") or a["label"],
            "group": groups.get(a.get("group"), {}).get("label", ""),
            "group_id": a.get("group", ""),
            "group_code": groups.get(a.get("group"), {}).get("code", ""),
            "question": a.get("question", ""),
            "subjects": [r["id"] for r in a.get("rows", [])],
        }
        for a in taxonomy.framework.get("areas", [])
    ]
    return {
        "version": taxonomy.framework.get("version", ""),
        "explorer": "https://mlsysbook.ai/taxonomy/",
        "areas": areas,
        "subjects": subjects,
        "concepts": concepts,
    }
