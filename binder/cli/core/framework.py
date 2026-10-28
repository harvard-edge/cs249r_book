"""Compile the optional candidate subject framework and its explicit book links.

A single candidate home is an editorial maintenance rule, not an assertion of
exclusive membership, a SKOS broader relation, completeness, or competence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import html
import io
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml

from .concept_graph import PAGE_URL, SITE_DIR, ConceptGraphError
from .field_graph import load_taxonomy_yaml
from .taxonomy_release import release_artifacts


DIMENSIONS = ("activities", "layers", "concerns", "contexts")
# Scope notes cite the row or area that owns a topic as [[identity]], so a
# relabeled row cannot leave a stale pointer behind.
SCOPE_REF = re.compile(r"\[\[([^\[\]]+)\]\]")
REQUIRED_DIMENSIONS = set(DIMENSIONS) - {"contexts"}
SUPPORTED_VERSIONS = ("0.1-draft", "0.2-draft", "0.3-draft")


def _fail(message):
    raise ConceptGraphError(f"framework: {message}")


def _text(record, names, context, allowed=None):
    if not isinstance(record, dict):
        _fail(f"{context} must be a mapping")
    # 2026-10-27: reject unknown keys so unquoted flow-mapping comma splits
    # (`run rules).: null`) cannot truncate text fields silently.
    if allowed is not None and set(record) - set(allowed):
        extra = ", ".join(sorted(set(record) - set(allowed)))
        _fail(f"{context} has unknown keys: {extra}")
    for name in names:
        if not isinstance(record.get(name), str) or not record[name].strip():
            _fail(f"{context} requires {name}")


def _records(values, context, nonempty=True):
    if not isinstance(values, list) or (nonempty and not values):
        _fail(f"{context} must be {'a nonempty' if nonempty else 'a'} list")
    seen = set()
    for item in values:
        _text(item, ("id",), context)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", item["id"]):
            _fail(f"invalid {context} ID {item['id']}")
        if item["id"] in seen:
            _fail(f"duplicate {context} ID {item['id']}")
        seen.add(item["id"])
    return values


def _members(values, known, context, nonempty=True):
    if not isinstance(values, list) or (nonempty and not values):
        _fail(f"{context} must be {'a nonempty' if nonempty else 'a'} list")
    if any(not isinstance(value, str) for value in values):
        _fail(f"{context} requires string IDs")
    if len(set(values)) != len(values):
        _fail(f"duplicate {context} value")
    if any(value not in known for value in values):
        _fail(f"unknown {context} value")


def _scope_notes(record, source_ids, context):
    for name in ("display_label", "warrant_role", "inventory_status"):
        if name in record:
            _text(record, (name,), context)
    for name in ("includes", "excludes"):
        if name in record:
            values = record[name]
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not value.strip() for value in values)):
                _fail(f"{context} {name} must be a nonempty text list")
    if "warrants" in record:
        _members(record["warrants"], source_ids, f"{context} warrants")


def _chapters(root: Path, codes, context):
    """Resolve `vN/NN` chapter codes to the chapter folder, title, and clean URL.

    A chapter pointer says where a row's subject is taught at chapter grain. It
    is an editorial reading pointer, not a passage-level alignment or a claim of
    teaching depth.
    """
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes) or len(set(codes)) != len(codes):
        _fail(f"{context} book_chapters must be a list of distinct codes")
    resolved = []
    for code in codes:
        match = re.fullmatch(r"v(?:ol)?([1-4])/(\d{2})", code)
        if not match:
            _fail(f"{context} has malformed chapter code {code}")
        volume = f"vol{match[1]}"
        # A chapter folder holds its own QMD; leftover build-output folders do not.
        folders = sorted(
            folder for folder in (root / "books" / volume).glob(f"{match[2]}_*/")
            if (folder / f"{folder.name}.qmd").exists()
        )
        if len(folders) != 1:
            _fail(f"{context} chapter code {code} does not resolve to one chapter")
        folder = folders[0]
        source = folder / f"{folder.name}.qmd"
        title_match = re.search(r"^#\s+(.+?)(?:\s+\{.*)?$", source.read_text(), re.M) if source.exists() else None
        resolved.append({"code": f"{volume}/{match[2]}", "volume": volume, "chapter": f"{volume}/{folder.name}",
                         "title": title_match[1].strip() if title_match else folder.name,
                         "url": f"https://mlsysbook.ai/{volume}/{folder.name}/{folder.name}.html"})
    return resolved


def load_framework(root: Path, field: dict) -> dict | None:
    """Validate optional authored rows; resolve only existing passage links.

    New field identities deliberately block compilation until their candidate
    home is considered. No classification or additional book link is guessed.
    """
    path = root / "taxonomy/ml-systems-framework.yml"
    if not path.exists():
        return None
    model = load_taxonomy_yaml(path)
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        _fail("unsupported schema")
    _text(model, ("id", "title", "version", "status", "scope"), "metadata",
          {"schema_version", "id", "title", "version", "status", "scope", "rationale", "row_rule",
           "secondary_policy", "relationship_semantics", "context_policy", "book_mapping_policy",
           "warrant_policy", "competency_policy", "tagging_rules", "placement_examples",
           "sources", "dimensions", "groups", "areas", "gaps"})
    if model["status"] != "editorial-draft" or model["version"] not in SUPPORTED_VERSIONS:
        _fail(f"only versions {', '.join(SUPPORTED_VERSIONS)} with editorial-draft status are supported")
    sources = _records(model.get("sources"), "source")
    for source in sources:
        _text(source, ("title", "url", "role"), source["id"], {"id", "title", "url", "role"})
        url = urlparse(source["url"])
        if (url.scheme != "https" or not url.netloc or url.username or url.password
                or re.search(r'[\s<>"{}|\\^`]', source["url"])):
            _fail(f"unsafe source URL {source['id']}")
    if len({s["title"].strip().casefold() for s in sources}) != len(sources):
        _fail("duplicate source title")
    dimensions = model.get("dimensions")
    if not isinstance(dimensions, dict) or not REQUIRED_DIMENSIONS <= set(dimensions) or set(dimensions) - set(DIMENSIONS):
        _fail("dimensions must define activities, layers, and concerns")
    facets = {}
    for name in DIMENSIONS:
        records = _records(dimensions.get(name, []), name, nonempty=name != "contexts")
        for record in records:
            _text(record, ("label",), record["id"], {"id", "label"})
        if len({r["label"].strip().casefold() for r in records}) != len(records):
            _fail(f"duplicate {name} label")
        facets[name] = {r["id"] for r in records}
    facet_labels = {r["label"].strip().casefold() for records in dimensions.values() for r in records}
    source_ids = {source["id"] for source in sources}
    concepts = {n["id"]: n for n in field["concepts"]}
    if len(concepts) != len(field["concepts"]):
        _fail("duplicate field concept identity")
    areas = _records(model.get("areas"), "area")
    row_ids, labels, homes = set(), set(), Counter()
    identities = {model["id"], *concepts}
    # Groups are an optional top tier (e.g. stack, lifecycle, qualities). When
    # present, every area names exactly one group; groups carry no concepts.
    groups = _records(model["groups"], "group") if "groups" in model else []
    for group in groups:
        _text(group, ("label", "question"), group["id"], {"id", "code", "label", "question"})
        if group["id"] in identities:
            _fail(f"group identity overlaps another source {group['id']}")
        identities.add(group["id"])
    group_ids = {group["id"] for group in groups}
    # Readable codes (L4.3) are optional but, once used, must be complete,
    # unique, and nested: an area code extends its group code, a subject code
    # its area code. Permanent identities stay the machine reference.
    if any("code" in area for area in areas):
        codes = set()
        records = [*groups, *model["areas"], *(row for area in model["areas"] for row in area.get("rows", []))]
        for record in records:
            code = record.get("code")
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z](\d+(\.\d+)?)?", code) or code in codes:
                _fail(f"{record.get('id')} requires a unique readable code")
            codes.add(code)
        for area in model["areas"]:
            parent = next((g["code"] for g in groups if g["id"] == area.get("group")), None)
            if parent and not re.fullmatch(re.escape(parent) + r"\d+", area["code"]):
                _fail(f"area code {area['code']} does not extend group code {parent}")
            for row in area.get("rows", []):
                if not re.fullmatch(re.escape(area["code"]) + r"\.\d+", row["code"]):
                    _fail(f"subject code {row['code']} does not extend area code {area['code']}")
    resolved_areas = []
    for area in areas:
        _text(area, ("label", "question", "scope"), area["id"],
              {"id", "code", "group", "label", "display_label", "question", "scope", "status",
               "includes", "excludes", "warrants", "warrant_role", "inventory_status", "rows"})
        _scope_notes(area, source_ids, area["id"])
        if groups and area.get("group") not in group_ids:
            _fail(f"area {area['id']} requires a known group")
        if not groups and "group" in area:
            _fail(f"area {area['id']} names a group but no groups are defined")
        if area["id"] in identities:
            _fail(f"area identity overlaps another source {area['id']}")
        identities.add(area["id"])
        key = area["label"].strip().casefold()
        if key in labels:
            _fail(f"duplicate area or row label {area['label']}")
        labels.add(key)
        rows = _records(area.get("rows"), "row")
        resolved_rows = []
        for row in rows:
            _text(row, ("label", "definition"), row["id"],
                  {"id", "code", "label", "display_label", "definition", "status", "includes", "excludes",
                   "warrants", "warrant_role", "inventory_status", "concepts", "secondary_concepts",
                   "book_chapters", *DIMENSIONS})
            _scope_notes(row, source_ids, row["id"])
            if row["id"] in identities or row["id"] in row_ids:
                _fail(f"duplicate or overlapping row identity {row['id']}")
            row_ids.add(row["id"])
            key = row["label"].strip().casefold()
            if key in labels:
                _fail(f"duplicate area or row label {row['label']}")
            if key in facet_labels:
                _fail(f"row label duplicates a facet label {row['label']}")
            labels.add(key)
            if row.get("status", "draft") != "draft" or area.get("status", "draft") != "draft":
                _fail("area and row assertions must remain draft")
            _members(row.get("concepts"), concepts, f"row {row['id']} concepts", nonempty=False)
            secondary = row.get("secondary_concepts", [])
            _members(secondary, concepts, f"row {row['id']} secondary concepts", nonempty=False)
            if set(secondary) & set(row["concepts"]):
                _fail(f"secondary concepts overlap own home in row {row['id']}")
            for name in DIMENSIONS:
                _members(row.get(name, [] if name == "contexts" else None), facets[name], f"row {row['id']} {name}", nonempty=name != "contexts")
            homes.update(row["concepts"])
            chapters = _chapters(root, row.get("book_chapters", []), f"row {row['id']}")
            members = set(row["concepts"]) | set(secondary)
            links = [deepcopy(link) for link in field.get("book_links", []) if link["concept"] in members]
            resolved_rows.append({**row, "contexts": row.get("contexts", []), "secondary_concepts": secondary,
                                  "book_chapters": chapters,
                                  "concept_count": len(row["concepts"]), "home_concept_count": len(row["concepts"]),
                                  "secondary_concept_count": len(secondary), "book_links": links,
                                  "inventory_status": row.get("inventory_status", "draft-inventory" if row["concepts"] else "unpopulated-candidate"),
                                  "book_link_count": len(links),
                                  "linked_concept_count": len({link["concept"] for link in links}),
                                  "book_volumes": sorted({link["volume"] for link in links})})
        resolved_areas.append({**area, "rows": resolved_rows})
    if row_ids & {area["id"] for area in areas}:
        _fail("area and row identities overlap")
    targets = row_ids | {area["id"] for area in areas}
    for area in resolved_areas:
        for record in (area, *area["rows"]):
            for note in (*record.get("includes", []), *record.get("excludes", [])):
                unknown = sorted(set(SCOPE_REF.findall(note)) - targets)
                if unknown:
                    _fail(f"{record['id']} scope note cites unknown row or area {', '.join(unknown)}")
    empty_groups = sorted(group_ids - {area.get("group") for area in areas})
    if empty_groups:
        _fail(f"groups without areas: {', '.join(empty_groups)}")
    duplicate = sorted(ident for ident, count in homes.items() if count != 1)
    if duplicate:
        _fail(f"concepts require exactly one candidate home: {', '.join(duplicate)}")
    unassigned = sorted(set(concepts) - set(homes))
    if unassigned:
        _fail(f"unassigned field concepts require a candidate home decision: {', '.join(unassigned)}")
    # 2026-10-27: taxonomy/README.md and .claude/rules/taxonomy.md
    # claim at least two relationships per concept and a connected graph.
    if model["version"] not in ("0.1-draft", "0.2-draft"):
        adj = {ident: set() for ident in concepts}
        for edge in field.get("relationships", []):
            adj[edge["source"]].add(edge["target"])
            adj[edge["target"]].add(edge["source"])
        under = sorted(ident for ident, neighbors in adj.items() if len(neighbors) < 2)
        if under:
            _fail(f"concepts require at least two relationships: {', '.join(under)}")
        seen, stack = set(), [next(iter(concepts))] if concepts else []
        while stack:
            curr = stack.pop()
            if curr not in seen:
                seen.add(curr)
                stack.extend(adj[curr] - seen)
        if len(seen) != len(concepts):
            _fail(f"concept graph is disconnected: {', '.join(sorted(set(concepts) - seen))}")
    gaps = _records(model.get("gaps"), "gap", nonempty=False)
    resolved_gaps = []
    rows_by_id = {row["id"]: row for area in resolved_areas for row in area["rows"]}
    for gap in gaps:
        _text(gap, ("reason", "status"), gap["id"],
              {"id", "topic", "label", "subject_row", "reason", "status"})
        if "subject_row" in gap:
            _text(gap, ("subject_row",), gap["id"])
            if gap["subject_row"] not in rows_by_id:
                _fail(f"unknown gap subject_row {gap['subject_row']}")
            resolved_gaps.append({**gap, "label": rows_by_id[gap["subject_row"]]["label"]})
        else:
            _text(gap, ("label",), gap["id"])
            resolved_gaps.append(deepcopy(gap))
        if gap["status"] != "candidate":
            _fail("gaps require candidate status, not a verified absence claim")
        if gap["id"] in identities or gap["id"] in row_ids:
            _fail(f"gap identity overlaps another source {gap['id']}")
    links = field.get("book_links", [])
    guides = _guides(root, rows_by_id, field)
    return {**deepcopy(model), "guides": guides, "dimensions": {name: dimensions.get(name, []) for name in DIMENSIONS}, "areas": resolved_areas, "gaps": resolved_gaps,
            "summary": {"groups": len(groups), "areas": len(areas), "rows": len(row_ids), "concepts": len(concepts),
                        "book_links": len(links), "linked_concepts": len({link["concept"] for link in links}),
                        "gaps": len(gaps), "roles": len(guides["roles"]), "symptoms": len(guides["symptoms"]),
                        "learning_paths": len(guides["learning_paths"])}}


def _guides(root: Path, rows_by_id: dict, field: dict) -> dict:
    """Load the reader guides (roles, symptoms, learning paths) and derive symptom levers.

    Every reference must name an existing subject or concept, and every symptom check or lever must
    share a relationship with one of the symptom's causes, so the guide stays tied to the graph (2026-10-28).
    """
    path = root / "taxonomy/ml-systems-guides.yml"
    if not path.exists():
        return {"roles": [], "symptoms": [], "learning_paths": []}
    model = load_taxonomy_yaml(path)
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        _fail("guides: unsupported schema")
    _text(model, (), "guides", {"schema_version", "roles_note", "roles", "symptoms_note", "symptoms", "paths_note", "learning_paths"})
    concepts = {c["id"]: c for c in field["concepts"]}
    roles = _records(model.get("roles"), "role")
    for role in roles:
        _text(role, ("label", "summary"), role["id"], {"id", "label", "summary", "core", "working"})
        _members(role.get("core"), rows_by_id, f"{role['id']} core")
        _members(role.get("working", []), rows_by_id, f"{role['id']} working", nonempty=False)
        if set(role["core"]) & set(role.get("working", [])):
            _fail(f"{role['id']} lists a subject as both core and working")
    edges = field.get("relationships", [])
    neighbors = {}
    for e in edges:
        neighbors.setdefault(e["source"], set()).add(e["target"])
        neighbors.setdefault(e["target"], set()).add(e["source"])
    symptoms = _records(model.get("symptoms"), "symptom")
    for symptom in symptoms:
        _text(symptom, ("label", "signal"), symptom["id"], {"id", "label", "signal", "causes", "checks", "levers"})
        _members(symptom.get("causes"), concepts, f"{symptom['id']} causes")
        _members(symptom.get("checks", []), concepts, f"{symptom['id']} checks", nonempty=False)
        _members(symptom.get("levers"), concepts, f"{symptom['id']} levers")
        for kind in ("checks", "levers"):
            for cid in symptom.get(kind, []):
                if not neighbors.get(cid, set()) & set(symptom["causes"]):
                    _fail(f"{symptom['id']} {kind[:-1]} {cid} shares no relationship with any listed cause")
    paths = _records(model.get("learning_paths"), "learning path")
    for lp in paths:
        _text(lp, ("label", "audience"), lp["id"], {"id", "label", "audience", "steps"})
        if not isinstance(lp.get("steps"), list) or len(lp["steps"]) < 3:
            _fail(f"{lp['id']} needs at least three steps")
        for step in lp["steps"]:
            _text(step, ("subject", "why"), lp["id"], {"subject", "why"})
            if step["subject"] not in rows_by_id:
                _fail(f"{lp['id']} step names unknown subject {step['subject']}")
        if len({step["subject"] for step in lp["steps"]}) != len(lp["steps"]):
            _fail(f"{lp['id']} repeats a subject")
    return {"roles_note": model.get("roles_note", ""), "roles": deepcopy(roles),
            "symptoms_note": model.get("symptoms_note", ""), "symptoms": deepcopy(symptoms),
            "paths_note": model.get("paths_note", ""), "learning_paths": deepcopy(paths)}


def _md(value):
    return html.escape(value, quote=False).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("[", "\\[").replace("]", "\\]").replace("*", "\\*").replace("_", "\\_").replace("#", "\\#")


def framework_artifacts(root: Path, field: dict, framework: dict | None) -> dict[Path, bytes]:
    """Export one deterministic table, reference, and data payload; no new claims."""
    if framework is None:
        return {}
    concepts = {c["id"]: c for c in field["concepts"]}
    facets = {name: {r["id"]: r["label"] for r in framework["dimensions"][name]} for name in DIMENSIONS}
    payload = {"schema_version": 1, "url": PAGE_URL, "framework": framework, "concepts": field["concepts"],
               "book_links": field.get("book_links", []), "relationships": field.get("relationships", []),
               "relations": field.get("relations", []), "retired": field.get("retired", [])}
    stream = io.StringIO(newline="")
    columns = ["framework_id", "framework_title", "framework_scope", "version", "status", "source_file",
               "group_id", "group_label", "area_id", "area_label", "area_display_label", "area_question", "area_scope", "row_id", "row_label", "row_display_label", "definition",
               "includes", "excludes", "warrant_source_ids", "warrant_source_titles", "warrant_source_urls", "warrant_role", "inventory_status",
               *DIMENSIONS, "concept_ids", "secondary_concept_ids", "concept_count", "home_concept_count", "secondary_concept_count", "book_link_count", "linked_concept_count", "book_volumes"]
    sources_by_id = {source["id"]: source for source in framework["sources"]}
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    lines = [f'# {_md(framework["title"])}', '', f'Version: `{framework["version"]}`. **Editorial draft.**', '',
             _md(framework["scope"]), '',
             'Each concept has one candidate home for editorial maintenance. All facet placements are candidate editorial synthesis; empty contexts mean relevance has not been assessed. This does not assert exclusive membership or a broader/narrower hierarchy. Passage links do not certify coverage, teaching depth, prerequisites, or competence.', '',
             '## Dimensions', '']
    for name in DIMENSIONS:
        lines.append(f'- **{name.title()}:** ' + '; '.join(_md(r["label"]) for r in framework["dimensions"][name]) + '.')
    groups = {group["id"]: group for group in framework.get("groups", [])}
    titles = {area["id"]: area["label"] for area in framework["areas"]}
    titles.update({row["id"]: row["label"] for area in framework["areas"] for row in area["rows"]})

    def note(value):
        return _md(SCOPE_REF.sub(lambda match: titles[match[1]], value))
    current_group = None
    for area in framework["areas"]:
        group = groups.get(area.get("group"))
        if group and group["id"] != current_group:
            current_group = group["id"]
            lines.extend(['', f'## {_md(group["label"])}', '', _md(group["question"])])
        heading = "###" if groups else "##"
        lines.extend(['', f'{heading} {_md(area["label"])}', '', _md(area["question"]), '', _md(area["scope"])])
        for row in area["rows"]:
            writer.writerow({"framework_id": framework["id"], "framework_title": framework["title"], "framework_scope": framework["scope"],
                             "version": framework["version"], "status": framework["status"], "source_file": "taxonomy/ml-systems-framework.yml",
                             "group_id": area.get("group", ""), "group_label": group["label"] if group else "",
                             "area_id": area["id"], "area_label": area["label"], "area_display_label": area.get("display_label", ""),
                             "area_question": area["question"], "area_scope": area["scope"],
                             "row_id": row["id"], "row_label": row["label"], "row_display_label": row.get("display_label", ""),
                             "includes": json.dumps(row.get("includes", []), ensure_ascii=False),
                             "excludes": json.dumps(row.get("excludes", []), ensure_ascii=False),
                             "warrant_source_ids": ";".join(row.get("warrants", [])),
                             "warrant_source_titles": json.dumps([sources_by_id[ident]["title"] for ident in row.get("warrants", [])], ensure_ascii=False),
                             "warrant_source_urls": json.dumps([sources_by_id[ident]["url"] for ident in row.get("warrants", [])]),
                             "warrant_role": row.get("warrant_role", ""), "inventory_status": row["inventory_status"],
                             "definition": row["definition"], **{name: ";".join(row[name]) for name in DIMENSIONS},
                             "concept_ids": ";".join(row["concepts"]), "secondary_concept_ids": ";".join(row["secondary_concepts"]),
                             **{name: row[name] for name in ("concept_count", "home_concept_count", "secondary_concept_count", "book_link_count", "linked_concept_count")},
                             "book_volumes": ";".join(row["book_volumes"])})
            lines.extend(['', f'{heading}# {_md(row["label"])}', '', f'Identity: `{row["id"]}`', '', _md(row["definition"]), ''])
            for name in DIMENSIONS:
                lines.append(f'- **{name.title()}:** ' + '; '.join(_md(facets[name][ident]) for ident in row[name]) + '.')
            for name in ("includes", "excludes"):
                if name in row:
                    lines.extend(['', f'{name.title()}: ' + '; '.join(note(value) for value in row[name]) + '.'])
            if row.get("warrants"):
                lines.extend(['', 'Design warrants: ' + ', '.join(f'`{ident}`' for ident in row["warrants"]) + '.'])
            lines.extend(['', 'Candidate home concepts:', ''])
            if not row["concepts"]:
                lines.append('No current field concept has this candidate home. The row records proposed subject scope, not evidence of absence from the books.')
            lines.extend(f'- **{_md(concepts[ident]["label"])}** (`{ident}`): {_md(concepts[ident]["definition"])}' for ident in row["concepts"])
            if row["secondary_concepts"]:
                lines.extend(['', 'Secondary concept connections (not additional homes):', ''])
                lines.extend(f'- **{_md(concepts[ident]["label"])}** (`{ident}`)' for ident in row["secondary_concepts"])
            lines.extend(['', 'Related MLSysBook passages (teaching depth not assessed):', ''])
            if not row["book_links"]:
                lines.append('No passage link is recorded. This is not evidence of absence from the books.')
            else:
                lines.extend(f'- {_md(link.get("volume", ""))}: {_md(link.get("chapter_title", ""))} — {_md(link.get("label", link["book_concept"]))} (`{link["book_concept"]}`)' for link in row["book_links"])
    lines.extend(['', '## Candidate scope gaps', '', 'Candidates for editorial investigation, not verified omissions.'])
    for gap in framework["gaps"]:
        topic = f'{_md(gap["topic"])}, in ' if gap.get("topic") else ''
        lines.extend(['', f'- **{topic}{_md(gap["label"])}** (`{gap["id"]}`): {_md(gap["reason"])}'])
    lines.extend(['', '## Design references', '', 'These references inform design and scope; they are not assertion-level validation.', ''])
    lines.extend(f'- [{_md(s["title"])}]({s["url"]}): {_md(s["role"])}' for s in framework["sources"])
    site = root / SITE_DIR
    return {**release_artifacts(root, framework, field),
            site / "framework-data.json": (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
            site / "framework-table.csv": stream.getvalue().encode(),
            site / "framework-reference.md": ("\n".join(lines) + "\n").encode()}
