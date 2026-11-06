"""Compile the ML systems taxonomy from its single source of truth (`taxonomy/taxonomy.yml`).

`taxonomy/taxonomy.yml` is the sole authored taxonomy file in the repository,
validated against `UnifiedTaxonomyModel` in `taxonomy/schema/mlsys-taxonomy.linkml.yml`.
`load_unified_taxonomy(root)` parses and validates that file once (cached by path and mtime)
and deterministically projects the four aspect models consumed by the compiler pipeline:
- `field`: concept inventory (`concepts`), `families`, `relations`, `edges`, `views` (with `views[id=all]`
  automatically derived from `[c["id"] for c in concepts]`).
- `framework`: subject `groups`, `areas`, `rows` (with each row's primary `concepts` list
  automatically derived in concept order from `concept["home"]`), multi-dimensional `dimensions`,
  and `gaps`.
- `guides`: practitioner `roles`, production `symptoms`, and `learning_paths`.
- `skos`: the optional polyhierarchical SKOS `topics`, `top_concepts`, `hierarchy`, and `collections` slice (authored under `skos_hierarchy:`).

Field identities never depend on chapter order or book ownership. External
reading references, editorial propositions, and book links (`taxonomy/mlsysbook-alignment.yml`)
have distinct roles; none is automatically promoted into reviewed coverage or a prerequisite.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import functools
import html
import json
import math
from pathlib import Path
import re

import yaml

from .concept_graph import SITE_DIR, ConceptGraphError, digest, svg_map

_BARE_SCI_SCALAR_RE = re.compile(r"^[+-]?[0-9]+(?:_[0-9]+)*(?:\.[0-9]*)?[eE][+-]?[0-9]+$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML SafeLoader that rejects duplicate keys and non-finite floats (including bare 1e999/-1e999 scalars)."""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConceptGraphError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _construct_finite_float(loader: _UniqueKeySafeLoader, node: yaml.ScalarNode) -> float:
    val = yaml.SafeLoader.construct_yaml_float(loader, node)
    if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        raise ConceptGraphError(f"non-finite YAML float {node.value!r} is forbidden")
    return float(val)


def _construct_checked_str(loader: _UniqueKeySafeLoader, node: yaml.ScalarNode) -> str:
    val = yaml.SafeLoader.construct_yaml_str(loader, node)
    # Plain (unquoted) YAML 1.1 scalars without a decimal dot such as `1e999` or `-1e999`
    # fall through to `tag:yaml.org,2002:str`; reject unquoted exponent-overflow literals while
    # preserving explicitly quoted string literals (`"1e999"`, `'1e999'`).
    if node.style in (None, "") and len(val) < 64 and _BARE_SCI_SCALAR_RE.match(val):
        try:
            num = float(val.replace("_", ""))
        except OverflowError as exc:
            raise ConceptGraphError(f"non-finite YAML exponent scalar {node.value!r} is forbidden") from exc
        if not math.isfinite(num):
            raise ConceptGraphError(f"non-finite YAML exponent scalar {node.value!r} is forbidden")
    return val


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
_UniqueKeySafeLoader.add_constructor(
    "tag:yaml.org,2002:float",
    _construct_finite_float,
)
_UniqueKeySafeLoader.add_constructor(
    "tag:yaml.org,2002:str",
    _construct_checked_str,
)


def load_taxonomy_yaml(path: Path) -> object:
    """Load a taxonomy file, refusing YAML anchors/aliases and duplicate mapping keys.

    Other tools (StaffML's hardened loader) reject aliases, and a merge script
    that reuses a list object silently emits them (2026-10-22), so an alias or
    duplicate mapping key here fails the build instead of breaking a downstream reader.
    """
    text = path.read_text()
    if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text, Loader=yaml.SafeLoader)):
        raise ConceptGraphError(f"{path.name}: YAML anchors/aliases are not allowed; write each value out in full")
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except ConceptGraphError as exc:
        raise ConceptGraphError(f"{path.name}: {exc}") from exc


TAXONOMY_FILE = "taxonomy/taxonomy.yml"


@functools.lru_cache(maxsize=4)
def _cached_unified_taxonomy(
    path_str: str, mtime_ns: int, schema_mtime_ns: int, root_str: str
) -> dict[str, dict | None] | None:
    """Parse, validate, and project the single-source taxonomy/taxonomy.yml."""
    path = Path(path_str)
    root = Path(root_str)
    raw = load_taxonomy_yaml(path)
    if not isinstance(raw, dict) or not ("areas" in raw and "concepts" in raw):
        return None
    if raw.get("schema_version") != 1:
        raise ConceptGraphError("taxonomy: unsupported schema")
    from .taxonomy_schema import (
        _cached_raw_schema,
        get_class_slots,
        resolve_schema_path,
        validate_linkml,
    )
    validate_linkml(raw, "UnifiedTaxonomyModel", root, TAXONOMY_FILE)
    schema_path = resolve_schema_path(root)
    schema_raw = _cached_raw_schema(str(schema_path), schema_mtime_ns or schema_path.stat().st_mtime_ns)
    if str(raw.get("version", "")) != str(schema_raw.get("version", "")):
        raise ConceptGraphError(
            f"schema sync: taxonomy version '{raw.get('version')}' disagrees with LinkML schema version '{schema_raw.get('version')}'"
        )
    framework_keys = get_class_slots("FrameworkModel", root)

    by_home: dict[str, list[str]] = {}
    concept_home: dict[str, str] = {}
    field_concepts: list[dict] = []
    for c in raw.get("concepts", []):
        if isinstance(c, dict):
            cid = c.get("id")
            home = c.get("home")
            if isinstance(cid, str):
                if cid in concept_home:
                    raise ConceptGraphError(f"field model: duplicate concepts ID {cid}")
                if isinstance(home, str):
                    concept_home[cid] = home
                    by_home.setdefault(home, []).append(cid)
            field_concepts.append({k: v for k, v in c.items() if k != "home"})
        else:
            field_concepts.append(c)

    rows_by_id: dict[str, dict] = {}
    for area in raw.get("areas", []):
        if isinstance(area, dict):
            for row in area.get("rows", []):
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    rid = row["id"]
                    if rid in rows_by_id:
                        raise ConceptGraphError(f"framework: duplicate or overlapping row identity {rid}")
                    rows_by_id[rid] = row
    known_rows = set(rows_by_id)
    unknown_homes = sorted(set(by_home) - known_rows)
    if unknown_homes:
        raise ConceptGraphError(f"framework: concept home names unknown subject row: {', '.join(unknown_homes)}")

    views = deepcopy(raw.get("views", []))
    if isinstance(views, list):
        if any(isinstance(v, dict) and v.get("id") == "all" for v in views):
            raise ConceptGraphError("taxonomy: views must not hand-list 'all'; it is derived automatically from concepts")
        views.append({
            "id": "all",
            "title": "The field landscape",
            "concepts": [c["id"] for c in field_concepts if isinstance(c, dict) and isinstance(c.get("id"), str)],
        })

    field_dict: dict = {
        "schema_version": raw.get("schema_version"),
        "id": raw.get("field_id", raw.get("id")),
        "title": raw.get("field_title", raw.get("title")),
        "scope": raw.get("field_scope", raw.get("scope")),
        "status": raw.get("status"),
        "sources": raw.get("field_sources", []),
        "families": raw.get("families", []),
        "concepts": field_concepts,
        "relationships": raw.get("relationships", []),
        "views": views,
    }
    for opt in ("background_boundary", "limitations", "learning_paths", "relations", "retired"):
        if opt in raw:
            field_dict[opt] = raw[opt]

    framework_dict: dict = {k: deepcopy(raw[k]) for k in framework_keys if k in raw}
    for area in framework_dict.get("areas", []):
        if isinstance(area, dict):
            for row in area.get("rows", []):
                if isinstance(row, dict):
                    if "concepts" in row:
                        raise ConceptGraphError(
                            f"taxonomy: subject row {row.get('id')} must not hand-list concepts; declare home on each concept"
                        )
                    row["concepts"] = list(by_home.get(row.get("id", ""), []))

    guides_raw = raw.get("guides")
    guides_dict = {"schema_version": raw["schema_version"], **deepcopy(guides_raw)} if isinstance(guides_raw, dict) else None
    skos_raw = raw.get("skos_hierarchy")
    skos_dict = {"schema_version": raw["schema_version"], **deepcopy(skos_raw)} if isinstance(skos_raw, dict) else None
    bundle = {"field": field_dict, "framework": framework_dict, "guides": guides_dict, "skos": skos_dict}
    enriched_concepts, _, _ = _validate_field_model(root, deepcopy(field_dict))
    from .framework import load_framework
    load_framework(root, {**field_dict, "concepts": enriched_concepts, "book_links": []}, _bundle=bundle)
    from .taxonomy import load_taxonomy
    load_taxonomy(root, {**field_dict, "concepts": enriched_concepts}, _bundle=bundle)
    return bundle


def load_unified_taxonomy(root: Path) -> dict[str, dict | None] | None:
    """Load the canonical single-source taxonomy/taxonomy.yml if present."""
    path = root / TAXONOMY_FILE
    if not path.exists():
        return None
    from .taxonomy_schema import resolve_schema_path
    schema_path = resolve_schema_path(root)
    schema_mtime = schema_path.stat().st_mtime_ns if schema_path.exists() else 0
    bundle = _cached_unified_taxonomy(str(path.resolve()), path.stat().st_mtime_ns, schema_mtime, str(root.resolve()))
    return deepcopy(bundle) if bundle is not None else None


def load_vocabulary(root: Path, context: str) -> tuple[set[str], dict[str, tuple[dict, dict]], dict[str, str]]:
    """Return `(concept_ids, rows_with_area, homes_by_concept)` from the unified taxonomy."""
    unified = load_unified_taxonomy(root)
    if unified is None:
        raise ConceptGraphError(f"{context}: unified taxonomy model is required")
    field = unified["field"]
    framework = unified["framework"]
    concepts = {c["id"] for c in field["concepts"]}
    rows: dict[str, tuple[dict, dict]] = {}
    homes: dict[str, str] = {}
    for area in framework["areas"]:
        for row in area["rows"]:
            rows[row["id"]] = (row, area)
            for cid in row.get("concepts", []):
                homes[cid] = row["id"]
    return concepts, rows, homes


def pct(part: int, whole: int) -> float:
    """Return `100 * part / whole` rounded to one decimal place (or `0.0` when `whole == 0`)."""
    return round(100 * part / whole, 1) if whole else 0.0


def _index(records, name):
    if not isinstance(records, list):
        raise ConceptGraphError(f"field model: {name} must be a list")
    result = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ConceptGraphError(f"field model: {name} requires string IDs")
        if record["id"] in result:
            raise ConceptGraphError(f"field model: duplicate {name} ID {record['id']}")
        result[record["id"]] = record
    return result


def _text(record, keys, context, allowed=None):
    if not isinstance(record, dict):
        raise ConceptGraphError(f"field model: {context} must be a mapping")
    # 2026-10-27: unquoted commas in flow mappings split strings into `key: null`
    # siblings that previously passed when required keys were still present.
    if allowed is not None and set(record) - set(allowed):
        extra = ", ".join(sorted(set(record) - set(allowed)))
        raise ConceptGraphError(f"field model: {context} has unknown keys: {extra}")
    for key in keys:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ConceptGraphError(f"field model: {context} requires {key}")


def _references(record, sources, context):
    refs = record.get("references", [])
    if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in sources for x in refs):
        unknown = [str(x) for x in (refs if isinstance(refs, list) else []) if x not in sources]
        detail = f": {', '.join(unknown)}" if unknown else ""
        raise ConceptGraphError(f"field model: {context} requires known reading references{detail}")


def _validate_field_model(root: Path, model: dict) -> tuple[list[dict], dict, dict]:
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise ConceptGraphError("field model: unsupported schema")
    from .taxonomy_schema import (
        CONCEPT_ID_RE,
        get_class_slots,
         get_enum_values,
        validate_field_semantics,
        validate_https_url,
        validate_linkml,
        validate_schema_sync,
    )
    _text(model, ("id", "title", "scope", "status"), "metadata", set(get_class_slots("FieldModel", root)))
    allowed_statuses = set(get_enum_values("ModelStatusEnum", root))
    if model["status"] not in allowed_statuses:
        raise ConceptGraphError(f"field model: unsupported status {model['status']}")
    sources = _index(model.get("sources"), "source")
    source_slots = set(get_class_slots("FieldSource", root))
    for ref in sources.values():
        _text(ref, ("title", "url", "kind", "scope"), ref["id"], source_slots)
        validate_https_url(ref["url"], f"field model: unsafe reading URL {ref['id']}")
    families = _index(model.get("families"), "family")
    scopes = set(get_enum_values("ConceptScopeEnum", root))
    relation_kinds = set(get_enum_values("RelationKindEnum", root))
    family_slots = set(get_class_slots("ConceptFamily", root))
    for family in families.values():
        _text(family, ("label", "scope", "question"), family["id"], family_slots)
        if family["scope"] not in scopes:
            raise ConceptGraphError(f"field model: unknown scope {family['scope']}")
        _references(family, sources, family["id"])
    concepts = _index(model.get("concepts"), "concept")
    concept_own_refs = set()
    concept_slots = set(get_class_slots("Concept", root)) - {"home"}
    for node in concepts.values():
        _text(node, ("label", "definition", "family", "status"), node["id"], concept_slots)
        if not CONCEPT_ID_RE.fullmatch(node["id"]) or node["family"] not in families:
            raise ConceptGraphError(f"field model: invalid concept identity or family {node['id']}")
        if node["status"] != "draft":
            raise ConceptGraphError("field model: a review label requires a future version-bound review contract")
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ConceptGraphError(f"field model: invalid aliases {node['id']}")
        if "references" in node:
            concept_own_refs.add(node["id"])
        node.setdefault("scope", families[node["family"]]["scope"])
        node.setdefault("references", list(families[node["family"]]["references"]))
        if node["scope"] not in scopes:
            raise ConceptGraphError(f"field model: unknown concept scope {node['id']}")
        _references(node, sources, node["id"])
    # A name looked up in the glossary or by a tagger must lead to one concept,
    # and a concept must not repeat its own label or alias in another case.
    # 2026-10-27: five names resolved to two concepts, and two concepts repeated
    # an alias with different capitalization.
    named = {}
    for node in concepts.values():
        for raw in [node["label"], *node.get("aliases", [])]:
            name = " ".join(raw.lower().split())
            if name in named:
                raise ConceptGraphError(f"field model: '{name}' names both {named[name]} and {node['id']}")
            named[name] = node["id"]
    # A controlled relation vocabulary (optional) gives every edge a verb with a
    # forward and inverse phrase, so readers can ask "what reduces X?". When the
    # vocabulary exists, every edge must use it and its kind must agree.
    relations = _index(model["relations"], "relation") if "relations" in model else {}
    relation_slots = set(get_class_slots("ControlledRelation", root))
    for relation in relations.values():
        _text(relation, ("forward", "inverse", "kind", "meaning"), relation["id"], relation_slots)
        if relation["kind"] not in relation_kinds:
            raise ConceptGraphError(f"field model: invalid relation kind {relation['id']}")
    # Concept identities are permanent: a retired identity is never reused and
    # names its replacement, so links and tags elsewhere keep resolving.
    retired_slots = set(get_class_slots("RetiredConcept", root))
    for record in model.get("retired", []):
        _text(record, ("id", "replaced_by", "version", "reason"), "retired concept", retired_slots)
        if record["id"] in concepts or record["replaced_by"] not in concepts:
            raise ConceptGraphError(f"field model: retired identity {record['id']} must be unused and name a current replacement")
    edges = _index(model.get("relationships"), "relationship")
    pairs = set()
    edge_slots = set(get_class_slots("Relationship", root))
    for edge in edges.values():
        _text(edge, ("source", "target", "predicate", "conditions", "basis", "status", "kind"), edge["id"], edge_slots)
        if "relations" in model:
            if edge.get("relation") not in relations:
                raise ConceptGraphError(f"field model: relationship needs a controlled relation {edge['id']}")
            if edge["kind"] != relations[edge["relation"]]["kind"]:
                raise ConceptGraphError(f"field model: kind disagrees with relation {edge['id']}")
        pair = frozenset((edge.get("source"), edge.get("target")))
        if pair in pairs:
            raise ConceptGraphError(f"field model: duplicate concept pair {edge['id']}")
        pairs.add(pair)
        if edge["source"] not in concepts or edge["target"] not in concepts or edge["source"] == edge["target"]:
            raise ConceptGraphError(f"field model: missing or identical endpoints {edge['id']}")
        if edge["basis"] != "editorial-synthesis" or edge["status"] != "draft":
            raise ConceptGraphError("field model: draft synthesis must not masquerade as independently verified evidence")
        if edge["kind"] not in relation_kinds:
            raise ConceptGraphError(f"field model: invalid relationship kind {edge['id']}")
        _references(edge, sources, edge["id"])
    views = _index(model.get("views"), "view")
    if "overview" not in views or "all" not in views:
        raise ConceptGraphError("field model: overview and all views are required")
    view_slots = set(get_class_slots("FieldView", root))
    for view in views.values():
        _text(view, ("title",), view["id"], view_slots)
        members = view.get("concepts", [])
        if not isinstance(members, list) or not members or len(set(members)) != len(members) or any(x not in concepts for x in members):
            unknown_v = [str(x) for x in (members if isinstance(members, list) else []) if x not in concepts]
            detail = f" (unknown: {', '.join(unknown_v[:5])})" if unknown_v else ""
            raise ConceptGraphError(f"field model: invalid view members {view['id']}{detail}")
    if set(views["all"]["concepts"]) != set(concepts):
        missing_all = sorted(set(concepts) - set(views["all"]["concepts"]))
        extra_all = sorted(set(views["all"]["concepts"]) - set(concepts))
        raise ConceptGraphError(
            f"field model: all view must retain every authored concept (missing: {missing_all[:5]}; extra: {extra_all[:5]})"
        )
    validate_linkml(model, "FieldModel", root, "field model")
    validate_field_semantics(model)
    validate_schema_sync(root, model)
    enriched_concepts = [
        {**node, "reading_scope": "concept" if node["id"] in concept_own_refs else "family"}
        for node in concepts.values()
    ]
    return enriched_concepts, concepts, edges


def load_field(root: Path, book_graph: dict) -> dict | None:
    """Validate authored identities and resolve only explicitly recorded links."""
    unified = load_unified_taxonomy(root)
    if unified is not None:
        model = unified["field"]
    else:
        path = root / TAXONOMY_FILE
        if not path.exists():
            return None
        model = load_taxonomy_yaml(path)
        if not isinstance(model, dict) or ("areas" in model or "topics" in model or "roles" in model):
            return None
    from .taxonomy_schema import get_class_slots, validate_linkml
    enriched_concepts, concepts, edges = _validate_field_model(root, model)

    alignment_path = root / "taxonomy/mlsysbook-alignment.yml"
    alignment = load_taxonomy_yaml(alignment_path) if alignment_path.exists() else {"schema_version": 1, "links": []}
    if not isinstance(alignment, dict) or alignment.get("schema_version") != 1 or not isinstance(alignment.get("links"), list):
        raise ConceptGraphError("field model: unsupported book-alignment schema")
    links, seen = [], set()
    link_slots = set(get_class_slots("BookAlignmentLink", root))
    for record in alignment["links"]:
        _text(record, ("concept", "book_concept", "status", "depth"), "book link", link_slots)
        key = (record["concept"], record["book_concept"])
        if key in seen or key[0] not in concepts or key[1] not in book_graph["nodes"]:
            raise ConceptGraphError(f"field model: duplicate or unresolved book link {key}")
        if record["status"] != "draft" or record["depth"] != "not-assessed":
            raise ConceptGraphError("field model: book links do not certify coverage or teaching depth")
        if record.get("relation", "related-passage") != "related-passage":
            raise ConceptGraphError("field model: book links do not establish conceptual equivalence")
        node = book_graph["nodes"][key[1]]
        if not node.get("section"):
            raise ConceptGraphError("field model: book links require a section-backed learner concept")
        chapter = book_graph["chapters"][node["chapter"]]
        links.append({**record, "label": node["label"], "chapter": node["chapter"],
                      "chapter_title": chapter["title"], "volume": f"vol{node['volume']}",
                      "url": node["url"], "section": node["section"],
                      "source": chapter["source"].relative_to(root).as_posix(),
                      "source_hash": digest(chapter["sections"][node["section"]])})
        seen.add(key)
    if alignment_path.exists():
        validate_linkml(alignment, "BookAlignmentModel", root, "field model")
    from .taxonomy import load_taxonomy
    taxonomy = load_taxonomy(root, model, _bundle=unified)
    from .framework import load_framework
    framework = load_framework(root, {**model, "concepts": enriched_concepts, "book_links": links}, _bundle=unified)
    degree = Counter(endpoint for edge in edges.values() for endpoint in (edge["source"], edge["target"]))
    return {**model, "concepts": enriched_concepts, "book_links": links, "taxonomy": taxonomy, "framework": framework,
            "summary": {"concepts": len(concepts), "relationships": len(edges), "book_links": len(links),
                        "linked_concepts": len({x["concept"] for x in links}),
                        "unconnected_concepts": [x for x in concepts if not degree[x]],
                        "by_scope": dict(Counter(x["scope"] for x in concepts.values()))}}


def field_artifacts(root: Path, book_graph: dict) -> dict[Path, bytes]:
    """Generate portable SVG and HTML plus website JSON from the same model."""
    field = load_field(root, book_graph)
    if field is None:
        return {}
    site = root / SITE_DIR
    nodes = {n["id"]: n for n in field["concepts"]}
    artifacts, diagrams, neighborhoods = {}, {}, {}
    render_views = list(field["views"])
    for ident in nodes:
        adjacent = {ident}
        for edge in field["relationships"]:
            if ident in (edge["source"], edge["target"]):
                adjacent.update((edge["source"], edge["target"]))
        members = [ident, *(n for n in nodes if n != ident and n in adjacent)]
        neighborhoods[ident] = members
        render_views.append({"id": f"focus-{ident}", "title": f"How does {nodes[ident]['label']} connect?", "concepts": members})
    for view in render_views:
        members = set(view["concepts"])
        data = {"title": view["title"],
                "nodes": [{"id": x, "label": nodes[x]["label"], "summary": nodes[x]["definition"]} for x in view["concepts"]],
                "edges": [{**e, "label": e["predicate"]} for e in field["relationships"] if e["source"] in members and e["target"] in members]}
        dot, svg = svg_map(data, pack_components=True, emphasize_first=False, pad_edge_labels=True, horizontal=view["id"] == "all")
        diagrams[view["id"]] = svg.decode()
        if view["id"].startswith("focus-"):
            artifacts[site / "field-focus" / f"{view['id'][6:]}.svg"] = svg
        else:
            artifacts[site / f"field-{view['id']}.svg"] = svg
            artifacts[site / f"field-{view['id']}.dot"] = dot.encode()
    payload = json.dumps({**field, "diagrams": diagrams, "neighborhoods": neighborhoods}, ensure_ascii=False, sort_keys=True, indent=2)
    artifacts[site / "field-data.json"] = (payload + "\n").encode()
    from .taxonomy import taxonomy_artifacts
    artifacts.update(taxonomy_artifacts(root, field, field.get("taxonomy")))
    from .framework import framework_artifacts
    artifacts.update(framework_artifacts(root, field, field.get("framework")))
    from .framework_figures import framework_figures
    if field.get("framework"):
        artifacts.update(framework_figures(root, field["framework"]))
    template = (site / "field.qmd").read_text().split("```{=html}\n", 1)[1].split("\n```", 1)[0]
    safe_payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>' + html.escape(field["title"]) + '</title><link rel="stylesheet" href="field.css">'
            '</head><body><script id="field-data" type="application/json">' + safe_payload + '</script>'
            + template + '</body></html>')
    artifacts[site / "field-preview.html"] = page.encode()
    return artifacts
