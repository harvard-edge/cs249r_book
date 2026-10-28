"""Validate an authored topic taxonomy and export its explicitly stated meaning.

Collections are navigation sets, not broader/narrower claims. Technical field
relationships and book links never become taxonomy or prerequisite statements.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import quote, urlparse

import yaml

from .concept_graph import PAGE_URL, SITE_DIR, ConceptGraphError


PREFIX = PAGE_URL


def _fail(message):
    raise ConceptGraphError(f"taxonomy: {message}")


def _text(record, names, context):
    if not isinstance(record, dict):
        _fail(f"{context} must be a mapping")
    for name in names:
        if not isinstance(record.get(name), str) or not record[name].strip():
            _fail(f"{context} requires {name}")


def _records(model, name):
    values = model.get(name)
    if not isinstance(values, list):
        _fail(f"{name} must be a list")
    seen = set()
    for value in values:
        _text(value, ("id",), name)
        if value["id"] in seen:
            _fail(f"duplicate {name} ID {value['id']}")
        seen.add(value["id"])
    return values


def load_taxonomy(root: Path, field: dict) -> dict | None:
    """Load the optional taxonomy without inferring classification or coverage."""
    path = root / "taxonomy/ml-systems-taxonomy.yml"
    if not path.exists():
        return None
    model = yaml.safe_load(path.read_text())
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        _fail("unsupported schema")
    _text(model, ("id", "title", "scope", "status"), "metadata")
    if model["status"] != "editorial-draft":
        _fail("only editorial-draft is supported before review provenance exists")
    sources = _records(model, "sources")
    for source in sources:
        _text(source, ("title", "url", "role"), source["id"])
        parsed = urlparse(source["url"])
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
                or re.search(r'[\s<>"{}|\\^`]', source["url"])):
            _fail(f"unsafe source URL {source['id']}")
    topics = _records(model, "topics")
    field_nodes = {n["id"]: n for n in field["concepts"]}
    topic_nodes = {n["id"]: n for n in topics}
    if set(topic_nodes) & set(field_nodes):
        _fail("topic IDs overlap field concepts")
    labels = {}
    for topic in topics:
        _text(topic, ("label", "definition", "status"), topic["id"])
        if not re.fullmatch(r"mlsys\.topic\.[a-z0-9-]+", topic["id"]):
            _fail(f"invalid topic identity {topic['id']}")
        if topic["status"] != "draft":
            _fail("topics require draft status")
    for node in [*field_nodes.values(), *topics]:
        label = node["label"].strip().casefold()
        if label in labels:
            _fail(f"duplicate concept label {node['label']} ({labels[label]}, {node['id']})")
        labels[label] = node["id"]
    nodes = {**field_nodes, **topic_nodes}
    hierarchy = model.get("hierarchy")
    if not isinstance(hierarchy, list):
        _fail("hierarchy must be a list")
    seen, parents = set(), {}
    for edge in hierarchy:
        _text(edge, ("narrower", "broader", "rationale", "status"), "hierarchy assertion")
        pair = edge["narrower"], edge["broader"]
        if pair[0] not in nodes or pair[1] not in topic_nodes:
            _fail(f"unknown or invalid hierarchy endpoints {pair}")
        if pair[0] == pair[1]:
            _fail("self hierarchy assertion")
        if pair in seen:
            _fail(f"duplicate hierarchy assertion {pair}")
        if edge["status"] != "draft":
            _fail("hierarchy assertions require draft status")
        seen.add(pair)
        parents.setdefault(pair[0], []).append(pair[1])
    visiting, visited = set(), set()

    def visit(ident):
        if ident in visiting:
            _fail(f"hierarchy cycle at {ident}")
        if ident in visited:
            return
        visiting.add(ident)
        for parent in parents.get(ident, []):
            visit(parent)
        visiting.remove(ident)
        visited.add(ident)

    for ident in nodes:
        visit(ident)
    roots = [n["id"] for n in topics if n["id"] not in parents]
    top_concepts = model.get("top_concepts")
    if (not isinstance(top_concepts, list) or
            any(not isinstance(x, str) for x in top_concepts) or
            len(set(top_concepts)) != len(top_concepts) or set(top_concepts) != set(roots)):
        _fail("top_concepts must equal root topic IDs")
    collections = _records(model, "collections")
    collection_ids = {c["id"] for c in collections}
    if collection_ids & (set(nodes) | {model["id"]}):
        _fail("collection IDs overlap concepts or scheme")
    if model["id"] in nodes:
        _fail("scheme ID overlaps a concept")
    families = {f["id"] for f in field["families"]}
    assigned, resolved_collections = set(), []
    for collection in collections:
        _text(collection, ("label", "scope_note", "family"), collection["id"])
        family = collection["family"]
        if family not in families or family in assigned:
            _fail(f"unknown or duplicate collection family {family}")
        if "members" in collection:
            _fail("collection members must derive from the authored field family")
        assigned.add(family)
        members = [ident for ident, node in field_nodes.items() if node["family"] == family]
        resolved_collections.append({**collection, "members": members})
    if assigned != families:
        _fail("collections must cover every field family exactly once")
    if any(node.get("family") not in assigned for node in field_nodes.values()):
        _fail("collections drop field concepts with unknown families")
    return {**model, "collections": resolved_collections,
            "roots": roots,
            "summary": {"topics": len(topics), "hierarchy_assertions": len(hierarchy),
                        "collections": len(collections), "field_concepts": len(field_nodes),
                        "classified_field_concepts": len(set(field_nodes) & set(parents))}}


def _uri(ident):
    return "<" + PREFIX + quote(ident, safe=".-_") + ">"


def _literal(value):
    # JSON strings are valid Turtle quoted strings, including control escaping.
    return json.dumps(value, ensure_ascii=False)


def taxonomy_artifacts(root: Path, field: dict, taxonomy: dict | None) -> dict[Path, bytes]:
    """Generate deterministic SKOS and a written reference from authored arrays."""
    if taxonomy is None:
        return {}
    nodes = {n["id"]: n for n in [*taxonomy["topics"], *field["concepts"]]}
    scheme = _uri(taxonomy["id"])
    ttl = ['@prefix skos: <http://www.w3.org/2004/02/skos/core#> .',
           '@prefix dcterms: <http://purl.org/dc/terms/> .', '',
           f'{scheme} a skos:ConceptScheme ;',
           f'  skos:prefLabel {_literal(taxonomy["title"])}@en ;',
           f'  skos:scopeNote {_literal(taxonomy["scope"])}@en ;',
           '  skos:editorialNote "Editorial draft; sources are design references, not assertion-level proof."@en .']
    for source in taxonomy["sources"]:
        ttl.append(f'{scheme} dcterms:source <{source["url"]}> .')
    for node in nodes.values():
        ttl.extend(['', f'{_uri(node["id"])} a skos:Concept ;',
                    f'  skos:inScheme {scheme} ;',
                    f'  skos:prefLabel {_literal(node["label"])}@en ;',
                    f'  skos:definition {_literal(node["definition"])}@en ;',
                    '  skos:editorialNote "Draft concept."@en .'])
        for alias in node.get("aliases", []):
            ttl.append(f'{_uri(node["id"])} skos:altLabel {_literal(alias)}@en .')
    for ident in taxonomy["roots"]:
        ttl.append(f'{_uri(ident)} skos:topConceptOf {scheme} .')
    for edge in taxonomy["hierarchy"]:
        ttl.append(f'{_uri(edge["narrower"])} skos:broader {_uri(edge["broader"])} .')
    for collection in taxonomy["collections"]:
        ttl.extend(['', f'{_uri(collection["id"])} a skos:Collection ;',
                    f'  skos:prefLabel {_literal(collection["label"])}@en ;',
                    f'  skos:scopeNote {_literal(collection["scope_note"])}@en .'])
        for ident in collection["members"]:
            ttl.append(f'{_uri(collection["id"])} skos:member {_uri(ident)} .')
    def md(value):
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("#", "\\#").replace("*", "\\*").replace("_", "\\_").replace("<", "&lt;").replace(">", "&gt;")
    lines = [f'# {md(taxonomy["title"])}', '', '**Editorial draft.** Classification is provisional; collection membership is not a broader/narrower assertion.', '', md(taxonomy["scope"]), '', '## Topic reference']
    for topic in taxonomy["topics"]:
        lines.extend(['', f'### {md(topic["label"])}', '', f'Identity: `{topic["id"]}`', '', md(topic["definition"])])
        parent_edges = [e for e in taxonomy["hierarchy"] if e["narrower"] == topic["id"]]
        if parent_edges:
            lines.extend(['', 'Broader topics: ' + ', '.join(md(nodes[e["broader"]]["label"]) for e in parent_edges) + '.'])
        children = [e for e in taxonomy["hierarchy"] if e["broader"] == topic["id"]]
        for edge in children:
            child = nodes[edge["narrower"]]
            lines.extend(['', f'- **{md(child["label"])}** (`{child["id"]}`): {md(edge["rationale"])}'])
    lines.extend(['', '## Navigation collections', '', 'These are authored groupings, not inferred taxonomic parents.'])
    for collection in taxonomy["collections"]:
        lines.extend(['', f'### {md(collection["label"])}', '', md(collection["scope_note"]), ''])
        lines.extend(f'- {md(nodes[ident]["label"])} (`{ident}`)' for ident in collection["members"])
    lines.extend(['', '## Concept definitions'])
    for node in field["concepts"]:
        lines.extend(['', f'### {md(node["label"])}', '', f'Identity: `{node["id"]}`', '', md(node["definition"])])
    site = root / SITE_DIR
    return {site / "taxonomy.ttl": ("\n".join(ttl) + "\n").encode(),
            site / "taxonomy-reference.md": ("\n".join(lines) + "\n").encode()}
