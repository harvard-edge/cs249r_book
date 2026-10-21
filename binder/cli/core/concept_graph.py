"""Compile chapter-owned learner graphs into maps and curriculum catalogs.

Book records live in chapter *_concepts.yml files. The independent field model
and book-link overlay live in shared/concepts. Published JSON, DOT, SVG, chapter
viewers, and the catalog are deterministic build artifacts.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from urllib.parse import quote as urlquote

import yaml


class ConceptGraphError(ValueError):
    """An authored graph cannot be resolved against its chapter contracts."""


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sections(text: str) -> dict[str, str]:
    """Index anchored headings, excluding executable and literal fenced blocks."""
    result = {}
    heading = re.compile(r"^(#{1,6})\s+.*?\{#([\w-]+)[^}]*\}.*$", re.M)
    # Preserve offsets but mask fences so code comments cannot become headings.
    masked = re.sub(
        r"^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$",
        lambda m: " " * len(m[0]),
        text,
        flags=re.M | re.S,
    )
    matches = list(heading.finditer(masked))
    for i, match in enumerate(matches):
        level = len(match[1])
        end = next(
            (m.start() for m in matches[i + 1 :] if len(m[1]) <= level), len(text)
        )
        if match[2] in result:
            raise ConceptGraphError(f"Duplicate section anchor: {match[2]}")
        result[match[2]] = text[match.start() : end]
    return result


def _required(record: dict, keys: tuple[str, ...], context: str) -> None:
    if not isinstance(record, dict):
        raise ConceptGraphError(f"{context}: expected a mapping")
    for key in keys:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ConceptGraphError(f"{context}: missing nonempty {key}")


def companion_resources(root: Path, chapter: dict, name: str) -> list[dict]:
    """Resolve explicitly authored activities; existence never implies readiness."""
    records = chapter["contract"].get("explorer_graph", {}).get("resources", {})
    if not isinstance(records, dict) or any(k not in chapter["owned"] for k in records):
        raise ConceptGraphError(f"{chapter['id']}: resources must reference owned concepts")
    items = records.get(name, [])
    if not isinstance(items, list):
        raise ConceptGraphError(f"{chapter['id']}: companion resources must be a list")
    result = []
    for record in items:
        _required(record, ("title", "path", "kind", "status"), chapter["id"])
        path = Path(record["path"])
        resolved = (root / path).resolve()
        if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            raise ConceptGraphError(f"{chapter['id']}: missing or unsafe companion source: {path}")
        if record["kind"] not in ("implementation", "simulation", "lab", "slides", "hardware"):
            raise ConceptGraphError(f"{chapter['id']}: unknown companion kind")
        if record["status"] not in ("early-release", "ready", "skeleton"):
            raise ConceptGraphError(f"{chapter['id']}: unknown companion status")
        result.append({**record, "url": "https://github.com/harvard-edge/cs249r_book/blob/dev/" + urlquote(path.as_posix(), safe="/")})
    return result


def load_graph(root: Path, volumes: tuple[str, ...] = ()) -> dict:
    """Resolve all nodes before relationships, allowing cross-chapter reuse."""
    chapters, nodes, edges, maps = {}, {}, {}, {}
    for number in range(1, 5):
        volume = f"vol{number}"
        if volumes and volume not in volumes:
            continue
        config = root / f"books/config/_quarto-pdf-{volume}.yml"
        order = config.read_text() if config.exists() else ""
        paths = list((root / "books" / volume).glob("*/*_concepts.yml"))
        paths.sort(
            key=lambda p: (
                order.find(f"{volume}/{p.parent.name}/")
                if f"{volume}/{p.parent.name}/" in order
                else len(order),
                p.parent.name,
            )
        )
        for path in paths:
            contract = yaml.safe_load(path.read_text())["concept_map"]
            source = path.parent / contract["source"]
            # Do not let authored paths escape their owning chapter.
            if source.resolve().parent != path.parent.resolve():
                raise ConceptGraphError(f"Source must be in owning chapter: {path}")
            text = source.read_text()
            indexed = sections(text)
            title_match = re.search(r"^#\s+(.+?)(?:\s+\{.*)?$", text, re.M)
            key = f"{volume}/{path.parent.name}"
            owned = contract.get("primary_concepts", []) + contract.get(
                "secondary_concepts", []
            )
            claims = contract.get("claims", {})
            title = title_match[1] if title_match else path.parent.name
            url = f"https://mlsysbook.ai/{volume}/{volume}/{path.parent.name}/{source.stem}.html"
            chapters[key] = dict(
                id=key,
                volume=volume,
                title=title,
                path=path,
                source=source,
                sections=indexed,
                contract=contract,
                owned=owned,
                url=url,
                source_hash=digest(text),
                maps=[],
            )
            graph = contract.get("learner_graph")
            if graph is None:
                continue
            if not isinstance(graph, dict) or graph.get("schema_version") != 1:
                raise ConceptGraphError(f"{path}: unsupported learner_graph schema")
            for node in graph.get("concepts", []):
                _required(node, ("id", "label", "teaches", "summary", "section"), key)
                aliases = node.get("aliases", [])
                if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
                    raise ConceptGraphError(f"{key}: concept aliases must be nonempty strings")
                ident = node["id"]
                if not re.fullmatch(
                    r"vol[1-4]\.[a-z0-9.-]+", ident
                ) or not ident.startswith(volume + "."):
                    raise ConceptGraphError(
                        f"Invalid volume-scoped concept ID: {ident}"
                    )
                if ident in nodes:
                    raise ConceptGraphError(f"Duplicate concept ID: {ident}")
                if node["teaches"] not in owned or node["teaches"] not in claims:
                    raise ConceptGraphError(
                        f"{ident}: teaches must name an owned concept with a claim"
                    )
                if node["section"] not in indexed:
                    raise ConceptGraphError(
                        f"{ident}: missing section {node['section']}"
                    )
                nodes[ident] = {
                    **node,
                    "chapter": key,
                    "volume": number,
                    "resources": companion_resources(root, chapters[key], node["teaches"]),
                    "claim": claims[node["teaches"]],
                    "concept": node["teaches"],
                    "url": url + "#" + node["section"],
                }
            for relation in graph.get("relationships", []):
                _required(
                    relation,
                    (
                        "id",
                        "source",
                        "target",
                        "predicate",
                        "explanation",
                        "conditions",
                        "status",
                    ),
                    key,
                )
                if relation["id"] in edges:
                    raise ConceptGraphError(
                        f"Duplicate relationship ID: {relation['id']}"
                    )
                if relation["status"] not in ("draft", "reviewed"):
                    raise ConceptGraphError(f"Unknown review status: {relation['id']}")
                evidence = relation.get("evidence", {})
                _required(evidence, ("section", "quote"), relation["id"])
                section = indexed.get(evidence["section"])
                if section is None:
                    raise ConceptGraphError(
                        f"{relation['id']}: missing evidence section"
                    )
                normalized = lambda s: " ".join(s.split())
                if normalized(evidence["quote"]) not in normalized(section):
                    raise ConceptGraphError(
                        f"{relation['id']}: evidence quote changed or is absent"
                    )
                edges[relation["id"]] = {
                    **relation,
                    "label": relation["predicate"],
                    "chapter": key,
                    "evidence_url": url + "#" + evidence["section"],
                    "evidence_hash": digest(section),
                }
            for view in graph.get("maps", []):
                _required(view, ("id", "focus_question", "description"), key)
                if view["id"] in maps or not re.fullmatch(
                    r"vol[1-4]\.[a-z0-9]+(?:[.-][a-z0-9]+)*", view["id"]
                ):
                    raise ConceptGraphError(
                        f"Duplicate or invalid map ID: {view['id']}"
                    )
                maps[view["id"]] = {
                    **view,
                    "chapter": key,
                    "title": view["focus_question"],
                }
                chapters[key]["maps"].append(view["id"])
    for edge in edges.values():
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise ConceptGraphError(
                f"{edge['id']}: relationship references missing concept"
            )
    for view in maps.values():
        members = view.get("concepts", [])
        relations = view.get("relationships", [])
        if not members or len(members) != len(set(members)):
            raise ConceptGraphError(
                f"{view['id']}: empty or repeated concept membership"
            )
        if not relations or len(relations) != len(set(relations)):
            raise ConceptGraphError(
                f"{view['id']}: empty or repeated relationship membership"
            )
        for ident in members:
            if ident not in nodes:
                raise ConceptGraphError(f"{view['id']}: missing concept {ident}")
        for ident in relations:
            if ident not in edges:
                raise ConceptGraphError(f"{view['id']}: missing relationship {ident}")
            edge = edges[ident]
            if edge["source"] not in members or edge["target"] not in members:
                raise ConceptGraphError(
                    f"{view['id']}: relationship endpoint outside map"
                )
        for check in view.get("checks", []):
            if (
                check.get("concept") not in members
                or not check.get("question")
                or not check.get("explanation")
            ):
                raise ConceptGraphError(f"{view['id']}: invalid learning check")
            choices = check.get("choices", [])
            if (
                len(choices) < 2
                or not isinstance(check.get("answer"), int)
                or not 0 <= check["answer"] < len(choices)
            ):
                raise ConceptGraphError(f"{view['id']}: invalid check answer")
    volume_views = {}
    for view in maps.values():
        scope = view.get("scope", "focused")
        if scope not in ("focused", "volume"):
            raise ConceptGraphError(f"{view['id']}: unknown authored map scope")
        volume = view["chapter"].split("/")[0]
        if any(not nodes[i]["chapter"].startswith(volume + "/") for i in view["concepts"]):
            raise ConceptGraphError(f"{view['id']}: map must stay inside its volume")
        if scope == "volume":
            if volume in volume_views:
                raise ConceptGraphError(f"{volume}: only one primary volume map is allowed")
            if view.get("volume") != volume:
                raise ConceptGraphError(f"{view['id']}: volume scope must name its owning volume")
            volume_views[volume] = view["id"]
    for chapter in chapters.values():
        if chapter["owned"] and chapter["volume"] in volume_views:
            ident = volume_views[chapter["volume"]]
            chapter["maps"] = [ident] + [m for m in chapter["maps"] if m != ident]
    result = dict(chapters=chapters, nodes=nodes, edges=edges, maps=maps)
    add_volume_maps(result, root)
    return result


def add_volume_maps(graph: dict, root: Path) -> None:
    """Join chapter-authored concept propositions into one navigable map per book.

    Evidence is the existing, editorially authored teaching claim. It establishes
    provenance and freshness; draft edges still require semantic review.
    """
    chapters, nodes, edges, maps = (
        graph[k] for k in ("chapters", "nodes", "edges", "maps")
    )
    for volume in sorted({c["volume"] for c in chapters.values()}):
        members, relations = set(), []

        def resolve(chapter_id: str, name: str) -> str:
            chapter = chapters.get(chapter_id)
            if (
                not chapter
                or not chapter_id.startswith(volume + "/")
                or name not in chapter["owned"]
            ):
                raise ConceptGraphError(
                    f"Volume map: unresolved concept {chapter_id}: {name}"
                )
            ident = volume + ".goal." + digest(chapter_id + ":" + name)[:16]
            alias = (
                chapter["contract"]
                .get("explorer_graph", {})
                .get("labels", {})
                .get(name, name)
            )
            nodes.setdefault(
                ident,
                dict(
                    id=ident,
                    label=alias,
                    summary=chapter["contract"]["claims"][name],
                    claim=chapter["contract"]["claims"][name],
                    concept=name,
                    chapter=chapter_id,
                    volume=int(volume[-1]),
                    url=chapter["url"],
                    kind="teaching-concept",
                    resources=companion_resources(root, chapter, name),
                ),
            )
            members.add(ident)
            return ident

        for chapter in chapters.values():
            if chapter["volume"] != volume:
                continue
            authored = chapter["contract"].get("explorer_graph", {})
            if not isinstance(authored.get("resources", {}), dict):
                raise ConceptGraphError(f"{chapter['id']}: resources must reference owned concepts")
            for goal in authored.get("resources", {}):
                companion_resources(root, chapter, goal)
            if authored and authored.get("schema_version") != 1:
                raise ConceptGraphError(
                    f"{chapter['id']}: unsupported explorer_graph schema"
                )
            for relation in authored.get("relationships", []):
                _required(
                    relation,
                    ("source", "predicate", "explanation", "conditions", "status"),
                    chapter["id"],
                )
                if relation["status"] not in ("draft", "reviewed"):
                    raise ConceptGraphError(
                        "Volume relationship status must be draft or reviewed"
                    )
                target = relation.get("target", {})
                _required(target, ("chapter", "concept"), chapter["id"])
                source_id = resolve(chapter["id"], relation["source"])
                target_id = resolve(volume + "/" + target["chapter"], target["concept"])
                evidence = relation.get("evidence", [])
                if not evidence:
                    raise ConceptGraphError("Volume relationship needs claim evidence")
                for item in evidence:
                    _required(item, ("chapter", "concept", "quote"), chapter["id"])
                    owner = chapters.get(volume + "/" + item["chapter"])
                    claim = (
                        owner["contract"].get("claims", {}).get(item["concept"], "")
                        if owner
                        else ""
                    )
                    if not claim or " ".join(item["quote"].split()) not in " ".join(
                        claim.split()
                    ):
                        raise ConceptGraphError(
                            f"{chapter['id']}: volume relationship evidence claim changed"
                        )
                ident = (
                    volume
                    + ".connection."
                    + digest(source_id + relation["predicate"] + target_id)[:16]
                )
                if ident in edges:
                    raise ConceptGraphError(f"Duplicate volume relationship: {ident}")
                edges[ident] = {
                    **relation,
                    "id": ident,
                    "source": source_id,
                    "target": target_id,
                    "label": relation["predicate"],
                    "chapter": chapter["id"],
                    "evidence_url": chapter["url"],
                    "evidence_kind": "chapter-claim",
                }
                relations.append(ident)
        if not relations:
            continue
        ident = volume + ".overview"
        title = {
            "vol1": "How do data, algorithms, and machines constrain one another?",
            "vol2": "How does a fleet turn capacity into useful work?",
            "vol3": "How does an agent turn proposals into accepted outcomes?",
            "vol4": "How does a physical system turn proposals into safe actions?",
        }[volume]
        # Membership follows canonical chapter and concept order, not edge traversal.
        ordered = [
            i
            for c in chapters.values()
            for name in c["owned"]
            if (i := volume + ".goal." + digest(c["id"] + ":" + name)[:16]) in members
        ]
        owner = next(c for c in chapters.values() if c["volume"] == volume)
        maps[ident] = dict(
            id=ident,
            title=title,
            focus_question=title,
            scope="argument" if any(m.get("scope") == "volume" and m["chapter"].startswith(volume + "/") for m in maps.values()) else "volume",
            volume=volume,
            chapter=owner["id"],
            description="Follow concepts across chapter boundaries. Chapter selection highlights a region of this same map.",
            concepts=ordered,
            relationships=relations,
        )
        for chapter in chapters.values():
            if chapter["volume"] == volume and chapter["owned"]:
                if maps[ident]["scope"] == "volume":
                    chapter["maps"].insert(0, ident)


def map_data(graph: dict, ident: str) -> dict:
    view = graph["maps"][ident]
    nodes = []
    for node_id in view["concepts"]:
        node = graph["nodes"][node_id]
        # Explicit teaches ownership joins overview arguments to finer questions.
        # This is a refinement/navigation link, never inferred equivalence.
        refinements = [
            {"id": candidate["id"], "title": candidate["title"]}
            for candidate in graph["maps"].values()
            if candidate.get("scope", "focused") == "focused" and candidate["id"] != ident
            and any(
                graph["nodes"][i]["chapter"] == node["chapter"]
                and graph["nodes"][i]["concept"] == node["concept"]
                for i in candidate["concepts"]
            )
        ]
        nodes.append({
            **node, "detail_maps": refinements,
            "teaching_goal": node["chapter"] + ":" + digest(node["concept"])[:16],
        })
    return {
        **view,
        "nodes": nodes,
        "coverage": map_coverage(graph, view),
        "edges": [graph["edges"][i] for i in view["relationships"]],
    }


def map_coverage(graph: dict, view: dict) -> dict:
    """Report editorial-goal representation, never claim field completeness."""
    volume = view["chapter"].split("/")[0]
    represented = {(graph["nodes"][i]["chapter"], graph["nodes"][i]["concept"]) for i in view["concepts"]}
    chapters = []
    for chapter in graph["chapters"].values():
        if chapter["volume"] != volume or not chapter["owned"]:
            continue
        covered = [name for name in chapter["owned"] if (chapter["id"], name) in represented]
        chapters.append(dict(
            id=chapter["id"], title=chapter["title"], represented=covered,
            outside_view=[name for name in chapter["owned"] if name not in covered],
            primary_outside_view=[name for name in chapter["contract"].get("primary_concepts", []) if name not in covered],
        ))
    return dict(
        teaching_goals=sum(len(c["owned"]) for c in graph["chapters"].values() if c["volume"] == volume),
        represented_goals=len(represented),
        teaching_chapters=len(chapters),
        represented_chapters=sum(bool(c["represented"]) for c in chapters),
        chapters=chapters,
    )


def svg_map(data: dict, *, pack_components: bool = False, emphasize_first: bool = True, pad_edge_labels: bool = False, horizontal: bool = False) -> tuple[str, bytes]:
    """Place labels and arrows together, retaining stable IDs for interaction."""
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        "digraph concepts {",
        'graph [rankdir=TB, bgcolor="white", pad="0.35", nodesep="0.45", ranksep="0.85", splines=polyline, outputorder=edgesfirst];',
        'node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", color="#88949d", penwidth=1.15, fontname="Helvetica Neue", fontsize=16, margin="0.18,0.12"];',
        'edge [fontname="Helvetica Neue", fontsize=12, color="#7a858e", fontcolor="#47535d", arrowsize=0.65];',
    ]
    if horizontal:
        lines.append("graph [rankdir=LR];")
    if pack_components:
        lines.append('graph [pack=true, packmode="array_u3"];')
    for i, node in enumerate(data["nodes"]):
        label = quote("\n".join(textwrap.wrap(node["label"], 23)))
        extra = ', fillcolor="#fcebed", color="#a51c30", penwidth=1.8' if i == 0 and emphasize_first else ""
        lines.append(
            f"{quote(node['id'])} [id={quote('concept-' + node['id'])}, label={label}, tooltip={quote(node['summary'])}{extra}];"
        )
    for edge in data["edges"]:
        if pad_edge_labels:
            words = "<BR/>".join(html.escape(line) for line in textwrap.wrap(edge["label"], 28))
            label = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5"><TR><TD>' + words + '</TD></TR></TABLE>>'
        else:
            label = quote("\n".join(textwrap.wrap(edge["label"], 23)))
        lines.append(
            f"{quote(edge['source'])} -> {quote(edge['target'])} [id={quote('relation-' + edge['id'])}, label={label}];"
        )
    dot = "\n".join(lines + ["}"]) + "\n"
    if not shutil.which("dot"):
        raise ConceptGraphError("Graphviz 'dot' is required to build maps")
    try:
        raw = subprocess.check_output(
            ["dot", "-Tsvg"], input=dot.encode(), stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise ConceptGraphError(exc.stderr.decode()) from exc
    root = ET.fromstring(raw)
    root.set("font-family", "Helvetica Neue, Helvetica, Arial, sans-serif")
    for text in root.findall(
        ".//{http://www.w3.org/2000/svg}g[@class='edge']/{http://www.w3.org/2000/svg}text"
    ):
        text.set("paint-order", "stroke")
        text.set("stroke", "white")
        text.set("stroke-width", "4")
        text.set("stroke-linejoin", "round")
    root.set("role", "group")
    root.set("aria-label", data["title"])
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    return dot, ET.tostring(root, encoding="utf-8", xml_declaration=True)


def curriculum_catalog(graph: dict, navigation: dict | None = None) -> dict:
    """Expose ownership and declared reuse without inventing semantic edges.

    Identities remain volume-scoped: similarly named concepts in independent
    textbooks are not automatically merged. Navigation lenses group chapters.
    """
    navigation = navigation or {}
    chapters = {}
    goals = {}
    lookup = {}
    for chapter in graph["chapters"].values():
        contract = chapter["contract"]
        concepts = []
        for kind, field in (
            ("core", "primary_concepts"),
            ("supporting", "secondary_concepts"),
        ):
            for name in contract.get(field, []):
                ident = chapter["id"] + ":" + digest(name)[:16]
                goal = dict(
                    id=ident,
                    label=name,
                    order=kind,
                    claim=contract.get("claims", {}).get(name, ""),
                    chapter=chapter["id"],
                    uses=[],
                )
                concepts.append(goal)
                goals[ident] = goal
                lookup[(chapter["id"], name)] = goal
        chapters[chapter["id"]] = {
            **{k: chapter[k] for k in ("id", "title", "url", "maps")},
            "concepts": concepts,
            "references": [],
        }
    for chapter in graph["chapters"].values():
        for field, relation in (("touches", "revisits"), ("previews", "previews")):
            for pointer in chapter["contract"].get(field, []):
                owner = chapter["volume"] + "/" + pointer["owner"]
                goal = lookup.get((owner, pointer["concept"]))
                if goal is None:
                    raise ConceptGraphError(
                        f"{chapter['id']}: unresolved {field} pointer: {pointer['concept']}"
                    )
                goal["uses"].append(dict(chapter=chapter["id"], relation=relation))
                chapters[chapter["id"]]["references"].append(
                    dict(concept=goal["id"], relation=relation)
                )
    themes = navigation.get("themes", [])
    seen = set()
    for theme in themes:
        if (
            not re.fullmatch(r"[a-z][a-z0-9-]*", theme.get("id", ""))
            or theme["id"] in seen
        ):
            raise ConceptGraphError("Invalid or duplicate atlas theme ID")
        seen.add(theme["id"])
        _required(theme, ("label", "question"), theme["id"])
        for ident in theme.get("chapters", []):
            if ident not in chapters:
                raise ConceptGraphError(
                    f"Atlas theme {theme['id']}: unknown chapter {ident}"
                )
    volumes = []
    for volume in sorted({c["volume"] for c in graph["chapters"].values()}):
        metadata = navigation.get("volumes", {}).get(volume, {})
        volumes.append(
            dict(
                id=volume,
                **metadata,
                chapters=[
                    ch for key, ch in chapters.items() if key.startswith(volume + "/")
                ],
            )
        )
    return dict(
        schema_version=2,
        scope="series",
        volumes=volumes,
        themes=themes,
        counts=dict(
            concepts=len(goals),
            chapters=len(chapters),
            references=sum(len(g["uses"]) for g in goals.values()),
        ),
        maps=[
            {k: view[k] for k in ("id", "title", "description", "chapter")}
            | {"scope": view.get("scope", "focused")}
            for view in sorted(
                (m for m in graph["maps"].values() if m.get("scope") != "argument"), key=lambda m: m.get("scope") != "volume"
            )
        ],
    )


def _json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build(root: Path, volumes: tuple[str, ...] = (), check: bool = False) -> dict:
    """Compile first, then publish artifacts; invalid authoring never overwrites outputs."""
    # Shared catalogs/maps always cover the series; volume flags scope chapter viewers.
    graph = load_graph(root)
    site = root / "site/explore"
    navigation_path = site / "atlas.yml"
    navigation = (
        yaml.safe_load(navigation_path.read_text()) if navigation_path.exists() else {}
    )
    catalog = curriculum_catalog(graph, navigation)
    artifacts = {site / "catalog.json": _json(catalog)}
    # The same portable viewer is served by the explorer and embedded chapters.
    template_path = site / "volume-maps.qmd"
    if not template_path.exists():
        template_path = site / "index.qmd"
    template_source = template_path.read_text()
    if "```{=html}\n" not in template_source:
        raise ConceptGraphError("Chapter viewers require a raw HTML volume-map template")
    template = template_source.split("```{=html}\n", 1)[1].split("\n```", 1)[0]
    if 'id="study-root"' in template:
        raise ConceptGraphError("Chapter viewers require the volume-map template, not the field explorer")
    for ident in graph["maps"]:
        data = map_data(graph, ident)
        dot, svg = svg_map(data)
        destination = site / "maps" / ident
        artifacts[destination / "graph.json"] = _json(data)
        artifacts[destination / "map.dot"] = dot.encode()
        artifacts[destination / "map.svg"] = svg
    for chapter in graph["chapters"].values():
        if not chapter["maps"] and not chapter["owned"]:
            continue
        if volumes and chapter["volume"] not in volumes:
            continue
        destination = chapter["source"].parent / "concepts-generated"
        # Chapter viewers start locally but retain the full ownership atlas.
        available_maps = [
            m["id"]
            for m in catalog["maps"]
            if m["id"] in graph["maps"]
        ]
        local_catalog = {
            **catalog,
            "scope": "chapter",
            "chapter": chapter["id"],
            "maps": [m for m in catalog["maps"] if m["id"] in available_maps],
        }
        artifacts[destination / "catalog.json"] = _json(local_catalog)
        for ident in available_maps:
            source = site / "maps" / ident
            for name in ("graph.json", "map.dot", "map.svg"):
                artifacts[destination / "maps" / ident / name] = artifacts[
                    source / name
                ]
        for name in (
            "explorer.js",
            "explorer.css",
            "atlas.js",
            "vendor/cytoscape.min.js",
            "vendor/LICENSE.cytoscape",
        ):
            artifacts[destination / name] = (site / name).read_bytes()
        # Inline generated data lets a chapter viewer open directly from disk as well.
        embedded = {
            "catalog": local_catalog,
            "maps": {
                ident: {
                    "graph": json.loads(
                        artifacts[site / "maps" / ident / "graph.json"]
                    ),
                    "svg": artifacts[site / "maps" / ident / "map.svg"].decode(),
                }
                for ident in available_maps
            },
        }
        payload = (
            json.dumps(embedded, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'
            + html.escape(chapter["title"])
            + ' concept maps</title><link rel="stylesheet" href="explorer.css"><style>body{margin:0;font-family:system-ui,sans-serif}.concept-explorer{padding:14px}.map-header h1{font-size:25px}.map-tools{gap:10px}</style></head><body>'
            + '<script type="application/json" id="concept-data">'
            + payload
            + "</script>"
            + template
            + "</body></html>"
        )
        artifacts[destination / "index.html"] = page.encode()
    manifest = {}
    for chapter in graph["chapters"].values():
        if chapter["maps"] or chapter["owned"]:
            manifest.setdefault(chapter["volume"], {})[
                chapter["source"].parent.name
            ] = dict(
                title=chapter["title"],
                viewer="concepts-generated/index.html",
                maps=chapter["maps"],
            )
    artifacts[root / "books/_build/concepts/chapters.json"] = _json(manifest)
    for volume in {c["volume"] for c in graph["chapters"].values()}:
        artifacts[site / "volumes" / f"{volume}.json"] = _json(
            next(v for v in catalog["volumes"] if v["id"] == volume)
        )
    series_payload = json.dumps(
        {
            "catalog": catalog,
            "maps": {
                ident: {
                    "graph": json.loads(
                        artifacts[site / "maps" / ident / "graph.json"]
                    ),
                    "svg": artifacts[site / "maps" / ident / "map.svg"].decode(),
                }
                for ident in graph["maps"] if any(m["id"] == ident for m in catalog["maps"])
            },
        },
        ensure_ascii=False,
    )
    series_payload = (
        series_payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    artifacts[site / "atlas.html"] = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>MLSysBook Concept Atlas</title><link rel="stylesheet" href="explorer.css">'
        "<style>body{margin:0;font-family:system-ui,sans-serif;background:#fafbfc}</style></head><body>"
        '<script type="application/json" id="concept-data">'
        + series_payload
        + "</script>"
        + template
        + "</body></html>"
    ).encode()
    # The field reference model has independent identities; book links are an overlay.
    from .field_graph import field_artifacts
    artifacts.update(field_artifacts(root, graph))
    stale = [
        str(p.relative_to(root))
        for p, data in artifacts.items()
        if not p.exists() or p.read_bytes() != data
    ]
    if not check:
        for path, data in artifacts.items():
            if path.exists() and path.read_bytes() == data:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    field_payload = artifacts.get(site / "field-data.json")
    return dict(
        chapters=len(graph["chapters"]),
        concepts=len(graph["nodes"]),
        relationships=len(graph["edges"]),
        maps=len(graph["maps"]),
        stale=stale,
        field=json.loads(field_payload)["summary"] if field_payload else None,
    )


def refresh(root: Path, volumes: tuple[str, ...] = ()) -> dict:
    """Find definition candidates and changed evidence; never author accepted records.

    This deterministic inventory is deliberately local: no model or embedding
    service is invoked. Semantic relationship discovery remains future work.
    """
    report = dict(schema_version=1, candidates=[], changed_evidence=[], invalid=[])
    snapshot_path = root / "books/_build/concepts/evidence-snapshot.json"
    previous = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    current = {}
    contracts = {
        (p.parent.parent.name, p.parent.name): yaml.safe_load(p.read_text())[
            "concept_map"
        ]
        for p in (root / "books").glob("vol*/*/*_concepts.yml")
    }
    for number in range(1, 5):
        volume = f"vol{number}"
        if volumes and volume not in volumes:
            continue
        for path in sorted((root / "books" / volume).glob("*/*_concepts.yml")):
            contract = yaml.safe_load(path.read_text())["concept_map"]
            source = path.parent / contract["source"]
            text = source.read_text()
            indexed = sections(text)
            learner = contract.get("learner_graph", {})
            labels = {
                label.casefold()
                for n in learner.get("concepts", [])
                for label in [n["label"], *n.get("aliases", [])]
            }
            for match in re.finditer(
                r':::\s*\{#([^\s}]+)[^\n]*\.callout-definition[^\n]*title="([^"]+)"',
                text,
            ):
                if match[2].casefold() not in labels:
                    report["candidates"].append(
                        dict(
                            label=match[2],
                            source=str(source.relative_to(root)),
                            anchor=match[1],
                            kind="definition",
                            status="candidate",
                        )
                    )
            # Index definition tags expose vocabulary outside callout boxes.
            known = labels | {
                c["label"].casefold() for c in report["candidates"]
                if c["source"] == str(source.relative_to(root))
            }
            for match in re.finditer(r"\\index\{([^{}\n]+)!definition\}", text):
                label = match[1].split("!", 1)[0].strip()
                if label.casefold() in known:
                    continue
                anchored = [
                    (anchor, section) for anchor, section in indexed.items()
                    if match[0] in section
                ]
                if not anchored:
                    continue
                anchor = min(anchored, key=lambda item: len(item[1]))[0]
                report["candidates"].append(dict(
                    label=label, source=str(source.relative_to(root)), anchor=anchor,
                    kind="index-definition", status="candidate",
                ))
                known.add(label.casefold())
            for edge in learner.get("relationships", []):
                ident = edge["id"]
                section = indexed.get(edge["evidence"]["section"])
                if section is None or " ".join(
                    edge["evidence"]["quote"].split()
                ) not in " ".join((section or "").split()):
                    report["invalid"].append(
                        dict(id=ident, reason="Supporting section or quote changed")
                    )
                now = dict(
                    section_hash=digest(section or ""),
                    authored_hash=digest(json.dumps(edge, sort_keys=True)),
                )
                old = previous.get(ident)
                if isinstance(old, str):
                    old = dict(section_hash=old, authored_hash=now["authored_hash"])
                if old and old["authored_hash"] == now["authored_hash"]:
                    current[ident] = old
                    if old["section_hash"] != now["section_hash"]:
                        report["changed_evidence"].append(ident)
                else:
                    current[ident] = now
            for relation in contract.get("explorer_graph", {}).get("relationships", []):
                target = relation["target"]
                source_id = (
                    volume
                    + ".goal."
                    + digest(
                        volume + "/" + path.parent.name + ":" + relation["source"]
                    )[:16]
                )
                target_id = (
                    volume
                    + ".goal."
                    + digest(
                        volume + "/" + target["chapter"] + ":" + target["concept"]
                    )[:16]
                )
                ident = (
                    volume
                    + ".connection."
                    + digest(source_id + relation["predicate"] + target_id)[:16]
                )
                claims = []
                for item in relation.get("evidence", []):
                    claim = (
                        contracts.get((volume, item["chapter"]), {})
                        .get("claims", {})
                        .get(item["concept"], "")
                    )
                    claims.append(claim)
                    if not claim or " ".join(item["quote"].split()) not in " ".join(
                        claim.split()
                    ):
                        report["invalid"].append(
                            dict(id=ident, reason="Supporting teaching claim changed")
                        )
                now = dict(
                    section_hash=digest(json.dumps(claims)),
                    authored_hash=digest(json.dumps(relation, sort_keys=True)),
                )
                old = previous.get(ident)
                if (
                    isinstance(old, dict)
                    and old["authored_hash"] == now["authored_hash"]
                ):
                    current[ident] = old
                    if old["section_hash"] != now["section_hash"]:
                        report["changed_evidence"].append(ident)
                else:
                    current[ident] = now
    for volume in (volumes or tuple(f"vol{n}" for n in range(1, 5))):
        labels = {
            label.casefold()
            for (vol, _), contract in contracts.items() if vol == volume
            for n in contract.get("learner_graph", {}).get("concepts", [])
            for label in [n["label"], *n.get("aliases", [])]
        }
        for glossary in sorted((root / "books" / volume / "backmatter").glob("**/glossary.qmd")):
            text = glossary.read_text()
            for match in re.finditer(r"^\*\*([^*\n]+)\*\*\s*\n:\s+([^\n]+)", text, re.M):
                if match[1].casefold() not in labels:
                    report["candidates"].append(dict(
                        label=match[1], source=str(glossary.relative_to(root)),
                        quote=match[2], kind="glossary", status="candidate", volume=volume,
                    ))
    output = root / "books/_build/concepts"
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.json").write_bytes(_json(report))
    # A restricted refresh must retain the baseline for other volumes.
    retained = {
        key: value
        for key, value in previous.items()
        if volumes and not any(key.startswith(v + ".") for v in volumes)
    }
    snapshot_path.write_bytes(_json({**retained, **current}))
    return report
