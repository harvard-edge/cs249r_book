"""Public downloads and StaffML export derived from `taxonomy/taxonomy.yml`.

Every release artifact (CSV, JSON, SKOS Turtle, GraphML, BibTeX, and the StaffML
`mlsys-taxonomy.json` bundle) is generated from the same compiled framework and field
records projected from `taxonomy/taxonomy.yml`, so the outline, the concept
list, the SKOS vocabulary, the graph, and the StaffML app cannot disagree. Groups, areas,
and subjects are SKOS collections (browsing places, not broader concepts); concepts are
SKOS concepts linked to their home subject by membership and to each other by the
controlled relations.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from ..concept_graph import PAGE_URL, SITE_DIR
from ..field_graph import load_taxonomy_yaml
from ..taxonomy_warrant import compute_warrant
from .schema import SCHEMA_REL_PATH

NAME = "ML Systems Taxonomy"
AUTHOR = "Vijay Janapa Reddi"
# The taxonomy release (identifiers, labels, definitions, relationships, guides) is CC BY 4.0 so courses and
# companies can reuse it; the book prose keeps its own license (decided 2026-10-28).
SYMMETRIC_RELATIONS = {"alternative-to", "trades-off"}
LICENSE = {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
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


def staffml_labels_payload(framework: dict, field: dict) -> dict:
    """Build the compact StaffML taxonomy labels payload (`staffml/app/src/data/mlsys-taxonomy.json`)."""
    subjects_with_area = {
        row["id"]: (row, area)
        for area in framework.get("areas", [])
        for row in area.get("rows", [])
    }
    home = {
        c: rid
        for rid, (row, _) in subjects_with_area.items()
        for c in row.get("concepts", [])
    }
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
        for rid, (row, area) in subjects_with_area.items()
    }
    concepts = {
        c["id"]: {"label": c["label"], "subject": home.get(c["id"], "")}
        for c in field.get("concepts", [])
    }
    groups = {g["id"]: g for g in framework.get("groups", [])}
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
        for a in framework.get("areas", [])
    ]
    return {
        "version": framework.get("version", ""),
        "explorer": PAGE_URL,
        "areas": areas,
        "subjects": subjects,
        "concepts": concepts,
    }


def release_artifacts(root: Path, framework: dict, field: dict) -> dict[Path, bytes]:
    """Generate the public release downloads (`taxonomy/downloads/*`) and StaffML labels bundle (`staffml/app/src/data/mlsys-taxonomy.json`).

    Validates both compiled JSON documents (`TaxonomyReleaseDocument` and `StaffMLLabelsDocument`)
    against `taxonomy/schema/mlsys-taxonomy.linkml.yml`.
    """
    out = root / SITE_DIR / "downloads"
    version = framework["version"]
    rows_by_id = {row["id"]: row for area in framework.get("areas", []) for row in area.get("rows", [])}
    outline = _outline(framework, field)
    concepts = outline["concepts"]
    home = {}
    for group in outline["groups"]:
        for area in group["areas"]:
            for subject in area["subjects"]:
                for cid in subject["concepts"]:
                    home[cid] = (group, area, subject)
    code_of = {s["id"]: s["code"] for g in outline["groups"] for a in g["areas"] for s in a["subjects"]}
    # Where the books discuss each concept (section-level candidate links, depth not assessed).
    passages = {}
    for link in field.get("book_links", []):
        passages.setdefault(link["concept"], []).append(
            {"volume": link.get("volume", ""), "chapter": link.get("chapter_title", ""), "url": link.get("url", "")})
    relationships = field.get("relationships", [])
    relations = field.get("relations", [])
    warrant = compute_warrant(root, field["concepts"], field.get("book_links", []))
    citation = (f"{AUTHOR}. {NAME}, version {version}. Machine Learning Systems, {YEAR}. {PAGE_URL}")
    status_val = str(framework.get("status", ""))
    is_draft = status_val == "editorial-draft" or str(version).endswith("-draft")
    is_rc = not is_draft and (status_val == "release-candidate" or "-rc" in str(version))
    if is_draft:
        base_note = "Editorial draft for expert review; not a claim of completeness or consensus."
    elif is_rc:
        base_note = "Release candidate under external governance review; not yet a normative release."
    else:
        base_note = "Externally governed normative release of the ML Systems Taxonomy."
    meta = {
        "name": NAME,
        "version": version,
        "status": framework["status"],
        "url": PAGE_URL,
        "license": LICENSE,
        "citation": citation,
        "note": base_note,
    }

    # JSON: the whole outline plus concepts and typed relationships.
    document = {**meta,
                "tagging_rules": framework.get("tagging_rules", []),
                "placement_examples": framework.get("placement_examples", []),
                "groups": outline["groups"],
                "concepts": [{"id": c["id"], "label": c["label"], "aliases": c.get("aliases", []), "definition": c["definition"],
                              "background": c.get("scope") == "background", "home": home[c["id"]][2]["code"] if c["id"] in home else "",
                              "home_id": home[c["id"]][2]["id"] if c["id"] in home else "",
                              **({"added_in": c["added_in"]} if c.get("added_in") else {}),
                              "warrant": warrant[c["id"]],
                              "book_passages": passages.get(c["id"], [])}
                             for c in field["concepts"]],
                "gaps": [{"id": g["id"], "topic": g.get("topic", ""), "subject": code_of.get(g.get("subject_row"), ""),
                          "status": g.get("status", ""), "reason": g["reason"]} for g in framework.get("gaps", [])],
                "relations": relations,
                "relationships": [{"source": r["source"], "relation": r.get("relation", ""), "target": r["target"],
                                   "statement": r.get("predicate", ""), "conditions": r.get("conditions", "")} for r in relationships],
                "retired": field.get("retired", []),
                "guides": framework.get("guides", {"roles": [], "symptoms": [], "learning_paths": []})}

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
    writer.writerow(["id", "term", "aliases", "definition", "subject_code", "subject", "area", "group", "background", "taught_in",
                     "added_in", "literary_warrant", "corpus_chapters", "organizational_warrant", "user_warrant_proxy"])
    for c in sorted(field["concepts"], key=lambda c: c["label"].casefold()):
        group, area, subject = home.get(c["id"], ({"label": ""}, {"label": ""}, {"code": "", "label": ""}))
        writer.writerow([c["id"], c["label"], "; ".join(c.get("aliases", [])), c["definition"], subject["code"], subject["label"],
                         area["label"], group["label"], "yes" if c.get("scope") == "background" else "no",
                         "; ".join(sorted({f'{p["volume"]}: {p["chapter"]}' for p in passages.get(c["id"], [])})),
                         c.get("added_in", ""), *(warrant[c["id"]][k] for k in ("literary", "corpus_chapters", "organizational", "user_proxy"))])

    # SKOS (Turtle).
    def lit(value):
        return json.dumps(str(value), ensure_ascii=False) + '@en'


    def iri(kind, ident):
        return f"<{PAGE_URL}{kind}/{ident}>"

    scheme = f"<{PAGE_URL}scheme>"
    prefixes = ["@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                f"@prefix taxmeta: <{PAGE_URL}scheme#> .", "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .", "@prefix dct: <http://purl.org/dc/terms/> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix dcat: <http://www.w3.org/ns/dcat#> .", "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
                f"@prefix mlsys: <{PAGE_URL}relation/> .", ""]
    # The extension vocabulary: one OWL property per controlled relation, under the SKOS property it refines.
    # `specializes` is the Z39.19 generic hierarchy (section 8.3.1), audited with the all-and-some test. `part-of` is
    # associative because section 8.3.3.2 sends parts of multiple wholes to the associative relationship.
    relation_ttl = prefixes + [f"<{PAGE_URL}relation/> a owl:Ontology ;", f"  dct:title {lit(NAME + ' relations')} ;",
                               f'  owl:versionInfo "{version}" ;', f"  dct:creator {lit(AUTHOR)} ;",
                               f"  dct:license <{LICENSE['url']}> ;", f"  owl:imports <http://www.w3.org/2004/02/skos/core> .", ""]
    for r in relations:
        parent = "skos:broader" if r["id"] == "specializes" else "skos:related"
        kinds = "owl:ObjectProperty, owl:SymmetricProperty" if r["id"] in SYMMETRIC_RELATIONS else "owl:ObjectProperty"
        relation_ttl += [f"mlsys:{r['id']} a {kinds} ;", f"  rdfs:subPropertyOf {parent} ;",
                         f"  rdfs:isDefinedBy <{PAGE_URL}relation/> ;", f"  rdfs:label {lit(r['forward'])} ;",
                         f"  rdfs:comment {lit(r['meaning'])} .", ""]
    relation_ttl += [
        "taxmeta:entryTerm a owl:DatatypeProperty ;",
        "  rdfs:domain skos:Concept ; rdfs:range rdf:langString ;",
        '  rdfs:label "search entry term"@en ;',
        '  rdfs:comment "Legacy retrieval term; may be a synonym, narrower term, or implementation name. Does not assert lexical equivalence."@en .', "",
        "taxmeta:condition a owl:DatatypeProperty ;",
        "  rdfs:domain rdf:Statement ; rdfs:range rdf:langString ;",
        '  rdfs:label "application condition"@en ;',
        '  rdfs:comment "Human-readable applicability condition for a described engineering relationship; not an executable logical rule."@en .', "",
    ]
    today_retired = {r["id"]: r for r in field.get("retired", [])}
    top = [c["id"] for c in field["concepts"]
           if not any(r["source"] == c["id"] and r.get("relation") == "specializes" for r in relationships)]
    ttl = prefixes + [f"{scheme} a skos:ConceptScheme ;", f"  dct:title {lit(NAME)} ;", f'  owl:versionInfo "{version}" ;',
                      f"  dct:creator {lit(AUTHOR)} ;", f"  dct:license <{LICENSE['url']}> ;",
                      f"  dct:description {lit(meta['note'])} ;", f"  dct:conformsTo <https://www.w3.org/TR/skos-reference/> ;",
                      f"  dct:requires <{PAGE_URL}relation/> ;",
                      "  skos:hasTopConcept " + ", ".join(iri("concept", c) for c in top) + " .", ""]

    def scope_note(s):
        parts = ["Includes: " + "; ".join(s["includes"])] if s.get("includes") else []
        parts += ["Excludes: " + "; ".join(s["excludes"])] if s.get("excludes") else []
        return " ".join(parts)

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
                note = scope_note(rows_by_id.get(s["id"], {}))
                ttl += [f"{iri('subject', s['id'])} a skos:Collection ;", f"  skos:prefLabel {lit(s['label'])} ;",
                        f'  skos:notation "{s["code"]}" ;', *([f"  skos:scopeNote {lit(note)} ;"] if note else []),
                        f"  skos:definition {lit(s['definition'])}" + (" ;" if members else " ."),
                        *([f"  skos:member {members} ."] if members else []), ""]
    provenance_path = root / "taxonomy/provenance.yml"
    term_records = load_taxonomy_yaml(provenance_path).get("concepts", {}) if provenance_path.exists() else {}
    for c in field["concepts"]:
        decisions = term_records.get(c["id"], {}).get("entry_term_decisions", {})
        equivalent = [a for a in c.get("aliases", []) if decisions.get(a, {}).get("kind") in {"synonym", "acronym"}]
        lines = [f"{iri('concept', c['id'])} a skos:Concept ;", f"  skos:inScheme {scheme} ;",
                 f"  skos:prefLabel {lit(c['label'])} ;", *[f"  taxmeta:entryTerm {lit(a)} ;" for a in c.get("aliases", [])],
                 *[f"  skos:altLabel {lit(a)} ;" for a in equivalent],
                 *([f"  skos:historyNote {lit('Added in ' + c['added_in'] + '.')} ;"] if c.get("added_in") else []),
                 *([f"  skos:topConceptOf {scheme} ;"] if c["id"] in top else []),
                 f"  skos:definition {lit(c['definition'])}"]
        for r in relationships:
            if r["source"] == c["id"] and r.get("relation"):
                lines[-1] += " ;"
                if r["relation"] == "specializes":
                    # Generic hierarchy is an all-instances claim. Preserve the
                    # materialized hierarchy for consumers without inference.
                    lines.append(f"  mlsys:specializes {iri('concept', r['target'])} ;")
                    lines.append(f"  skos:broader {iri('concept', r['target'])}")
                else:
                    # Association is safe to expose for navigation; the stronger
                    # typed engineering claim is described with its condition below.
                    lines.append(f"  skos:related {iri('concept', r['target'])}")
        ttl += lines[:-1] + [lines[-1] + " .", ""]
    # RDF 1.1 reification describes a statement without entailing its triple.
    # Do not additionally assert conditional typed triples: consumers could then
    # ignore the condition and infer a stronger claim than the source supports.
    for index, r in enumerate(relationships):
        # Legacy candidate graphs have wording but no controlled relation type.
        # Retain their wording with only a generic association predicate.
        predicate_iri = f"mlsys:{r['relation']}" if r.get("relation") else "skos:related"
        ttl += [f"_:statement{index} a rdf:Statement ;",
                f"  dct:identifier {lit(r['id'])} ;",
                f"  rdf:subject {iri('concept', r['source'])} ;",
                f"  rdf:predicate {predicate_iri} ;",
                f"  rdf:object {iri('concept', r['target'])} ;",
                f"  rdfs:label {lit(r.get('predicate', ''))} ;",
                f"  taxmeta:condition {lit(r.get('conditions', ''))} .", ""]
    for rid, r in today_retired.items():
        ttl += [f"{iri('concept', rid)} a skos:Concept ;", f"  skos:inScheme {scheme} ;", "  owl:deprecated true ;",
                *([f"  dct:isReplacedBy {iri('concept', r['replaced_by'])} ;"] if r.get("replaced_by") else []),
                f"  skos:prefLabel {lit('Retired: ' + rid)} ;", f"  skos:historyNote {lit(r.get('reason', 'Retired.'))} .", ""]

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

    bib_status_note = "Editorial draft" if is_draft else ("Release candidate" if is_rc else "Normative release")
    bib = (f"@misc{{mlsys_taxonomy_{version.replace('.', '_').replace('-', '_')},\n"
           f"  author = {{{AUTHOR}}},\n  title = {{{NAME}}},\n  version = {{{version}}},\n  year = {{{YEAR}}},\n"
           f"  howpublished = {{\\url{{{PAGE_URL}}}}},\n  note = {{{bib_status_note}. Licensed {LICENSE['name']}}}\n}}\n")

    result = {out / "mlsys-taxonomy.json": (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode(),
              out / "mlsys-taxonomy-subjects.csv": subjects.getvalue().encode(),
              out / "mlsys-taxonomy-concepts.csv": glossary.getvalue().encode(),
              out / "mlsys-taxonomy.ttl": ("\n".join(ttl) + "\n").encode(),
              out / "mlsys-concept-graph.graphml": ("\n".join(graph) + "\n").encode(),
              out / "mlsys-taxonomy.bib": bib.encode()}
    result[out / "mlsys-taxonomy-relations.ttl"] = ("\n".join(relation_ttl) + "\n").encode()
    result[out / "CITATION.cff"] = (
        "cff-version: 1.2.0\nmessage: If you use this taxonomy, please cite it using this metadata.\ntype: dataset\n"
        f'title: "{NAME}"\nversion: "{version}"\nauthors:\n  - family-names: Reddi\n    given-names: Vijay Janapa\n'
        f'url: "{PAGE_URL}"\nlicense: CC-BY-4.0\n').encode()
    formats = {"mlsys-taxonomy.ttl": "text/turtle", "mlsys-taxonomy-relations.ttl": "text/turtle",
               "mlsys-taxonomy.json": "application/json", "mlsys-taxonomy-concepts.csv": "text/csv",
               "mlsys-taxonomy-subjects.csv": "text/csv", "mlsys-concept-graph.graphml": "application/graphml+xml"}
    dcat = prefixes + [f"<{PAGE_URL}dataset> a dcat:Dataset ;", f"  dct:title {lit(NAME)} ;", f"  dct:description {lit(meta['note'])} ;",
                       f"  dct:creator {lit(AUTHOR)} ;", f"  dct:license <{LICENSE['url']}> ;", f'  dcat:version "{version}" ;',
                       f"  dcat:landingPage <{PAGE_URL}> ;",
                       "  dcat:distribution " + ", ".join(f"<{PAGE_URL}downloads/{name}>" for name in formats) + " .", ""]
    for name, media in formats.items():
        dcat += [f"<{PAGE_URL}downloads/{name}> a dcat:Distribution ;", f"  dcat:downloadURL <{PAGE_URL}downloads/{name}> ;",
                 f'  dcat:mediaType <https://www.iana.org/assignments/media-types/{media}> ;',
                 f"  dct:license <{LICENSE['url']}> .", ""]
    result[out / "mlsys-taxonomy.dcat.ttl"] = ("\n".join(dcat) + "\n").encode()

    # Resolvable identifiers: every concept, relation, and the scheme answers at its URI (GitHub Pages cannot content-negotiate,
    # so each URI is a small page that redirects to its entry and links the RDF). Generated, gitignored, staged for publication.
    def stub(path, title, target):
        result[root / SITE_DIR / path / "index.html"] = (
            f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{xml_escape(title)}</title>'
            f'<meta http-equiv="refresh" content="0; url={target}"><link rel="canonical" href="{PAGE_URL}{path}">'
            f'<link rel="alternate" type="text/turtle" href="{PAGE_URL}downloads/mlsys-taxonomy.ttl"></head>'
            f'<body><a href="{target}">{xml_escape(title)}</a></body></html>\n').encode()

    for c in field["concepts"]:
        stub(f"concept/{c['id']}", c["label"], f"{PAGE_URL}#concept={c['id']}")
    for rid, r in today_retired.items():
        stub(f"concept/{rid}", f"Retired: {rid}", f"{PAGE_URL}#concept={r.get('replaced_by', '')}")
    for r in relations:
        stub(f"relation/{r['id']}", r["forward"], f"{PAGE_URL}downloads/mlsys-taxonomy-relations.ttl")
    stub("relation", f"{NAME} relations", f"{PAGE_URL}downloads/mlsys-taxonomy-relations.ttl")
    stub("scheme", NAME, PAGE_URL)
    stub("dataset", NAME, f"{PAGE_URL}downloads/mlsys-taxonomy.dcat.ttl")
    linkml_src = root / SCHEMA_REL_PATH
    if linkml_src.exists():
        from .schema import compile_json_schema, validate_linkml
        validate_linkml(document, "TaxonomyReleaseDocument", root, "taxonomy release")
        result[out / "mlsys-taxonomy.linkml.yml"] = linkml_src.read_bytes()
        result[out / "mlsys-taxonomy.schema.json"] = (
            json.dumps(compile_json_schema(root, "TaxonomyReleaseDocument"), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
        staffml_data_dir = root / "staffml/app/src/data"
        if staffml_data_dir.is_dir():
            staffml_payload = staffml_labels_payload(framework, field)
            validate_linkml(staffml_payload, "StaffMLLabelsDocument", root, "staffml taxonomy labels")
            result[staffml_data_dir / "mlsys-taxonomy.json"] = (
                json.dumps(staffml_payload, indent=1, ensure_ascii=False) + "\n"
            ).encode()
    return result
