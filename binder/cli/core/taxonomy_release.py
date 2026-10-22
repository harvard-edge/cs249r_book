"""Public downloads of the ML systems taxonomy in standard formats.

Every file is generated from the same compiled framework and field records, so
the outline, the concept list, the SKOS vocabulary, and the graph cannot
disagree. Groups, areas, and subjects are SKOS collections (browsing places,
not broader concepts); concepts are SKOS concepts linked to their home subject
by membership and to each other by the controlled relations.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

BASE = "https://mlsysbook.ai/taxonomy/"
PAGE = "https://mlsysbook.ai/explore/"
NAME = "ML Systems Taxonomy"
AUTHOR = "Vijay Janapa Reddi"
LICENSE = {"name": "CC BY-NC-SA 4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"}
YEAR = "2026"


def _outline(framework: dict, field: dict) -> dict:
    concepts = {c["id"]: c for c in field["concepts"]}
    groups = {g["id"]: {**g, "areas": []} for g in framework.get("groups", [])}
    for area in framework["areas"]:
        subjects = [{
            "code": row.get("code", ""), "id": row["id"], "label": row["label"], "definition": row["definition"],
            "includes": row.get("includes", []), "excludes": row.get("excludes", []), "contexts": row.get("contexts", []),
            "concepts": row["concepts"], "related_concepts": row.get("secondary_concepts", []),
            "taught_in": [{"volume": c["volume"], "chapter": c["title"], "url": c["url"]} for c in row.get("book_chapters", [])],
        } for row in area["rows"]]
        record = {"code": area.get("code", ""), "id": area["id"], "label": area["label"], "question": area["question"],
                  "scope": area["scope"], "subjects": subjects}
        groups.setdefault(area.get("group", "all"), {"id": "all", "code": "", "label": "Areas", "question": "", "areas": []})["areas"].append(record)
    return {"groups": [{k: g[k] for k in ("code", "id", "label", "question", "areas") if k in g} for g in groups.values()],
            "concepts": concepts}


def release_artifacts(root: Path, framework: dict, field: dict) -> dict[Path, bytes]:
    out = root / "site/explore/downloads"
    version = framework["version"]
    outline = _outline(framework, field)
    concepts = outline["concepts"]
    home = {}
    for group in outline["groups"]:
        for area in group["areas"]:
            for subject in area["subjects"]:
                for cid in subject["concepts"]:
                    home[cid] = (group, area, subject)
    relationships = field.get("relationships", [])
    relations = field.get("relations", [])
    citation = (f"{AUTHOR}. {NAME}, version {version}. Machine Learning Systems, {YEAR}. {PAGE}")
    meta = {"name": NAME, "version": version, "status": framework["status"], "url": PAGE, "license": LICENSE,
            "citation": citation, "note": "Editorial draft for expert review; not a claim of completeness or consensus."}

    # JSON: the whole outline plus concepts and typed relationships.
    document = {**meta,
                "groups": outline["groups"],
                "concepts": [{"id": c["id"], "label": c["label"], "aliases": c.get("aliases", []), "definition": c["definition"],
                              "background": c.get("scope") == "background", "home": home[c["id"]][2]["code"] if c["id"] in home else ""}
                             for c in field["concepts"]],
                "relations": relations,
                "relationships": [{"source": r["source"], "relation": r.get("relation", ""), "target": r["target"],
                                   "statement": r.get("predicate", ""), "conditions": r.get("conditions", "")} for r in relationships]}

    # CSV: one row per subject (the outline) and one row per concept (the glossary).
    subjects = io.StringIO(newline="")
    writer = csv.writer(subjects, lineterminator="\n")
    writer.writerow(["code", "group", "area", "subject", "definition", "concepts", "concept_ids", "taught_in"])
    for group in outline["groups"]:
        for area in group["areas"]:
            for s in area["subjects"]:
                writer.writerow([s["code"], group["label"], area["label"], s["label"], s["definition"],
                                 "; ".join(concepts[c]["label"] for c in s["concepts"]), "; ".join(s["concepts"]),
                                 "; ".join(f'{t["volume"]}: {t["chapter"]}' for t in s["taught_in"])])
    glossary = io.StringIO(newline="")
    writer = csv.writer(glossary, lineterminator="\n")
    writer.writerow(["id", "term", "aliases", "definition", "subject_code", "subject", "area", "group", "background"])
    for c in sorted(field["concepts"], key=lambda c: c["label"].casefold()):
        group, area, subject = home.get(c["id"], ({"label": ""}, {"label": ""}, {"code": "", "label": ""}))
        writer.writerow([c["id"], c["label"], "; ".join(c.get("aliases", [])), c["definition"], subject["code"], subject["label"],
                         area["label"], group["label"], "yes" if c.get("scope") == "background" else "no"])

    # SKOS (Turtle).
    def lit(value):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"@en'

    def iri(kind, ident):
        return f"<{BASE}{kind}/{ident}>"

    ttl = ["@prefix skos: <http://www.w3.org/2004/02/skos/core#> .", "@prefix dct: <http://purl.org/dc/terms/> .",
           "@prefix owl: <http://www.w3.org/2002/07/owl#> .", f"@prefix mlsys: <{BASE}relation/> .", "",
           f"<{BASE}scheme> a skos:ConceptScheme ;", f"  dct:title {lit(NAME)} ;", f'  owl:versionInfo "{version}" ;',
           f"  dct:creator {lit(AUTHOR)} ;", f"  dct:license <{LICENSE['url']}> ;", f"  dct:description {lit(meta['note'])} .", ""]
    for r in relations:
        ttl += [f"mlsys:{r['id']} a owl:ObjectProperty ;", "  owl:subPropertyOf skos:related ;",
                f"  skos:prefLabel {lit(r['forward'])} ;", f"  skos:definition {lit(r['meaning'])} .", ""]
    for group in outline["groups"]:
        ttl += [f"{iri('group', group['id'])} a skos:Collection ;", f"  skos:prefLabel {lit(group['label'])} ;",
                f'  skos:notation "{group.get("code", "")}" ;',
                "  skos:member " + ", ".join(iri("area", a["id"]) for a in group["areas"]) + " .", ""]
        for area in group["areas"]:
            ttl += [f"{iri('area', area['id'])} a skos:Collection ;", f"  skos:prefLabel {lit(area['label'])} ;",
                    f'  skos:notation "{area["code"]}" ;', f"  skos:scopeNote {lit(area['scope'])} ;",
                    "  skos:member " + ", ".join(iri("subject", s["id"]) for s in area["subjects"]) + " .", ""]
            for s in area["subjects"]:
                members = ", ".join(iri("concept", c) for c in s["concepts"])
                ttl += [f"{iri('subject', s['id'])} a skos:Collection ;", f"  skos:prefLabel {lit(s['label'])} ;",
                        f'  skos:notation "{s["code"]}" ;', f"  skos:definition {lit(s['definition'])}" + (" ;" if members else " ."),
                        *([f"  skos:member {members} ."] if members else []), ""]
    for c in field["concepts"]:
        lines = [f"{iri('concept', c['id'])} a skos:Concept ;", f"  skos:inScheme <{BASE}scheme> ;",
                 f"  skos:prefLabel {lit(c['label'])} ;", *[f"  skos:altLabel {lit(a)} ;" for a in c.get("aliases", [])],
                 f"  skos:definition {lit(c['definition'])}"]
        for r in relationships:
            if r["source"] == c["id"] and r.get("relation"):
                lines[-1] += " ;"
                lines.append(f"  mlsys:{r['relation']} {iri('concept', r['target'])}")
        ttl += lines[:-1] + [lines[-1] + " .", ""]

    # GraphML for Gephi, yEd, Cytoscape desktop, and graph databases.
    keys = [("label", "node"), ("home", "node"), ("subject", "node"), ("area", "node"), ("group", "node"), ("background", "node"),
            ("relation", "edge"), ("statement", "edge"), ("conditions", "edge")]
    graph = ['<?xml version="1.0" encoding="UTF-8"?>', '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
             *[f'  <key id="{k}" for="{scope}" attr.name="{k}" attr.type="string"/>' for k, scope in keys],
             f'  <graph id="{xml_escape(NAME)} {xml_escape(version)}" edgedefault="directed">']
    for c in field["concepts"]:
        group, area, subject = home.get(c["id"], ({"label": ""}, {"label": ""}, {"code": "", "label": ""}))
        data = {"label": c["label"], "home": subject["code"], "subject": subject["label"], "area": area["label"],
                "group": group["label"], "background": "yes" if c.get("scope") == "background" else "no"}
        graph.append(f'    <node id="{xml_escape(c["id"])}">' + "".join(f'<data key="{k}">{xml_escape(str(v))}</data>' for k, v in data.items()) + "</node>")
    for i, r in enumerate(relationships):
        data = {"relation": r.get("relation", ""), "statement": r.get("predicate", ""), "conditions": r.get("conditions", "")}
        graph.append(f'    <edge id="e{i}" source="{xml_escape(r["source"])}" target="{xml_escape(r["target"])}">'
                     + "".join(f'<data key="{k}">{xml_escape(str(v))}</data>' for k, v in data.items()) + "</edge>")
    graph += ["  </graph>", "</graphml>"]

    bib = (f"@misc{{mlsys_taxonomy_{version.replace('.', '_').replace('-', '_')},\n"
           f"  author = {{{AUTHOR}}},\n  title = {{{NAME}}},\n  version = {{{version}}},\n  year = {{{YEAR}}},\n"
           f"  howpublished = {{\\url{{{PAGE}}}}},\n  note = {{Editorial draft. Licensed {LICENSE['name']}}}\n}}\n")

    return {out / "mlsys-taxonomy.json": (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(),
            out / "mlsys-taxonomy-subjects.csv": subjects.getvalue().encode(),
            out / "mlsys-taxonomy-concepts.csv": glossary.getvalue().encode(),
            out / "mlsys-taxonomy.ttl": ("\n".join(ttl) + "\n").encode(),
            out / "mlsys-concept-graph.graphml": ("\n".join(graph) + "\n").encode(),
            out / "mlsys-taxonomy.bib": bib.encode()}
