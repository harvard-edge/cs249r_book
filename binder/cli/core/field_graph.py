"""Compile a field reference model and an independent book-link overlay.

Field identities never depend on chapter order or book ownership. External
reading references, editorial propositions, and book links have distinct roles;
none is automatically promoted into reviewed coverage or a prerequisite.
"""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml

from .concept_graph import ConceptGraphError, digest, svg_map


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


def _text(record, keys, context):
    if not isinstance(record, dict):
        raise ConceptGraphError(f"field model: {context} must be a mapping")
    for key in keys:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ConceptGraphError(f"field model: {context} requires {key}")


def _references(record, sources, context):
    refs = record.get("references", [])
    if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in sources for x in refs):
        raise ConceptGraphError(f"field model: {context} requires known reading references")


def load_field(root: Path, book_graph: dict) -> dict | None:
    """Validate authored identities and resolve only explicitly recorded links."""
    path = root / "shared/concepts/ml-systems-field.yml"
    if not path.exists():
        return None
    model = yaml.safe_load(path.read_text())
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise ConceptGraphError("field model: unsupported schema")
    _text(model, ("id", "title", "scope", "status"), "metadata")
    if model["status"] != "editorial-draft":
        raise ConceptGraphError("field model: only editorial-draft is supported before review provenance exists")
    sources = _index(model.get("sources"), "source")
    for ref in sources.values():
        _text(ref, ("title", "url", "kind", "scope"), ref["id"])
        url = urlparse(ref["url"])
        if url.scheme != "https" or not url.netloc:
            raise ConceptGraphError(f"field model: unsafe reading URL {ref['id']}")
    families = _index(model.get("families"), "family")
    scopes = {"core", "background", "model-extension", "generative-extension", "agent-extension", "physical-extension", "specialist"}
    for family in families.values():
        _text(family, ("label", "scope", "question"), family["id"])
        if family["scope"] not in scopes:
            raise ConceptGraphError(f"field model: unknown scope {family['scope']}")
        _references(family, sources, family["id"])
    concepts = _index(model.get("concepts"), "concept")
    for node in concepts.values():
        _text(node, ("label", "definition", "family", "status"), node["id"])
        if not re.fullmatch(r"mlsys\.[a-z0-9-]+", node["id"]) or node["family"] not in families:
            raise ConceptGraphError(f"field model: invalid concept identity or family {node['id']}")
        if node["status"] != "draft":
            raise ConceptGraphError("field model: a review label requires a future version-bound review contract")
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ConceptGraphError(f"field model: invalid aliases {node['id']}")
        node.setdefault("scope", families[node["family"]]["scope"])
        node["reading_scope"] = "concept" if "references" in node else "family"
        node.setdefault("references", list(families[node["family"]]["references"]))
        if node["scope"] not in scopes:
            raise ConceptGraphError(f"field model: unknown concept scope {node['id']}")
        _references(node, sources, node["id"])
    # A controlled relation vocabulary (optional) gives every edge a verb with a
    # forward and inverse phrase, so readers can ask "what reduces X?". When the
    # vocabulary exists, every edge must use it and its kind must agree.
    relations = _index(model["relations"], "relation") if "relations" in model else {}
    for relation in relations.values():
        _text(relation, ("forward", "inverse", "kind", "meaning"), relation["id"])
        if relation["kind"] not in {"mechanism", "constraint", "tradeoff", "evaluation", "dependency", "failure", "composition"}:
            raise ConceptGraphError(f"field model: invalid relation kind {relation['id']}")
    # Concept identities are permanent: a retired identity is never reused and
    # names its replacement, so links and tags elsewhere keep resolving.
    for record in model.get("retired", []):
        _text(record, ("id", "replaced_by", "version", "reason"), "retired concept")
        if record["id"] in concepts or record["replaced_by"] not in concepts:
            raise ConceptGraphError(f"field model: retired identity {record['id']} must be unused and name a current replacement")
    edges = _index(model.get("relationships"), "relationship")
    pairs = set()
    for edge in edges.values():
        if relations:
            if edge.get("relation") not in relations:
                raise ConceptGraphError(f"field model: relationship needs a controlled relation {edge['id']}")
            if edge["kind"] != relations[edge["relation"]]["kind"]:
                raise ConceptGraphError(f"field model: kind disagrees with relation {edge['id']}")
        pair = frozenset((edge.get("source"), edge.get("target")))
        if pair in pairs:
            raise ConceptGraphError(f"field model: duplicate concept pair {edge['id']}")
        pairs.add(pair)
        _text(edge, ("source", "target", "predicate", "conditions", "basis", "status", "kind"), edge["id"])
        if edge["source"] not in concepts or edge["target"] not in concepts or edge["source"] == edge["target"]:
            raise ConceptGraphError(f"field model: missing or identical endpoints {edge['id']}")
        if edge["basis"] != "editorial-synthesis" or edge["status"] != "draft":
            raise ConceptGraphError("field model: draft synthesis must not masquerade as independently verified evidence")
        if edge["kind"] not in {"mechanism", "constraint", "tradeoff", "evaluation", "dependency", "failure", "composition"}:
            raise ConceptGraphError(f"field model: invalid relationship kind {edge['id']}")
        _references(edge, sources, edge["id"])
    views = _index(model.get("views"), "view")
    if "overview" not in views or "all" not in views:
        raise ConceptGraphError("field model: overview and all views are required")
    for view in views.values():
        _text(view, ("title",), view["id"])
        members = view.get("concepts", [])
        if not isinstance(members, list) or not members or len(set(members)) != len(members) or any(x not in concepts for x in members):
            raise ConceptGraphError(f"field model: invalid view members {view['id']}")
    if set(views["all"]["concepts"]) != set(concepts):
        raise ConceptGraphError("field model: all view must retain every authored concept")

    alignment_path = path.with_name("mlsysbook-alignment.yml")
    alignment = yaml.safe_load(alignment_path.read_text()) if alignment_path.exists() else {"schema_version": 1, "links": []}
    if not isinstance(alignment, dict) or alignment.get("schema_version") != 1 or not isinstance(alignment.get("links"), list):
        raise ConceptGraphError("field model: unsupported book-alignment schema")
    links, seen = [], set()
    for record in alignment["links"]:
        _text(record, ("concept", "book_concept", "status", "depth"), "book link")
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
    from .taxonomy import load_taxonomy
    taxonomy = load_taxonomy(root, model)
    from .framework import load_framework
    framework = load_framework(root, {**model, "book_links": links})
    degree = Counter(endpoint for edge in edges.values() for endpoint in (edge["source"], edge["target"]))
    return {**model, "book_links": links, "taxonomy": taxonomy, "framework": framework,
            "summary": {"concepts": len(concepts), "relationships": len(edges), "book_links": len(links),
                        "linked_concepts": len({x["concept"] for x in links}),
                        "unconnected_concepts": [x for x in concepts if not degree[x]],
                        "by_scope": dict(Counter(x["scope"] for x in concepts.values()))}}


def field_artifacts(root: Path, book_graph: dict) -> dict[Path, bytes]:
    """Generate portable SVG and HTML plus website JSON from the same model."""
    field = load_field(root, book_graph)
    if field is None:
        return {}
    site = root / "site/explore"
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
