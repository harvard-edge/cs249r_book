"""Reconcile declared chapter concepts without certifying field completeness."""
from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
import unicodedata

import yaml

from .concept_graph import ConceptGraphError, digest, load_graph
from .field_graph import load_field


DECISION_KEYS = {"inventory_id", "chapter", "label", "role", "disposition", "field_concepts", "rationale", "reviewer", "reviewer_kind", "reviewed_on", "source_hash", "field_hash", "evidence"}


DISPOSITIONS = {"mapped-to", "narrower-than", "broader-than", "candidate-new", "out-of-scope"}
TARGETED = {"mapped-to", "narrower-than", "broader-than"}


def _hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _normal(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def decision_hashes(chapter, label, role, field_nodes, decision):
    """Bind a disposition to its declaration, cited sections, and field targets."""
    declaration = dict(label=label, role=role, claim=chapter["contract"].get("claims", {}).get(label, ""))
    evidence = sorted({(e["section"], digest(chapter["sections"].get(e["section"], ""))) for e in decision.get("evidence", [])})
    source_hash = _hash(dict(declaration=declaration, evidence=evidence))
    targets = decision.get("field_concepts", [])
    if targets:
        content = [{k: field_nodes[t].get(k) for k in ("id", "label", "definition", "aliases", "family", "scope")} for t in sorted(targets) if t in field_nodes]
    else:
        content = sorted((c["id"], _normal(c["label"]), sorted(map(_normal, c.get("aliases", []))), c.get("definition"), c.get("scope"), c.get("family")) for c in field_nodes.values())
    return source_hash, _hash(content)


def coverage_report(root: Path, volumes: tuple[str, ...] = ()) -> dict:
    """Inventory every declaration; suggestions never become review decisions.

    Edits to cited sections, the declaration, or dependent field records reopen
    a decision. IDs match catalog goal IDs; renaming a label or chapter orphans
    its decision and requires explicit rebinding.
    """
    graph = load_graph(root)
    field = load_field(root, graph)
    if field is None:
        raise ConceptGraphError("coverage: field model is required")
    field_nodes = {c["id"]: c for c in field["concepts"]}
    path = root / "shared/concepts/coverage-ledger.yml"
    ledger = yaml.safe_load(path.read_text()) if path.exists() else {"schema_version": 1, "decisions": []}
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1 or not isinstance(ledger.get("decisions"), list):
        raise ConceptGraphError("coverage: invalid ledger schema")
    if set(ledger) - {"schema_version", "decisions"}:
        raise ConceptGraphError("coverage: unknown ledger keys")
    decisions = {}
    for decision in ledger["decisions"]:
        if not isinstance(decision, dict) or not isinstance(decision.get("inventory_id"), str):
            raise ConceptGraphError("coverage: decision requires inventory_id")
        ident = decision["inventory_id"]
        if set(decision) - DECISION_KEYS:
            raise ConceptGraphError(f"coverage: unknown decision keys {ident}")
        if ident in decisions:
            raise ConceptGraphError(f"coverage: duplicate decision {ident}")
        if decision.get("disposition") not in DISPOSITIONS:
            raise ConceptGraphError(f"coverage: invalid disposition {ident}")
        for name in ("rationale", "reviewer", "source_hash", "field_hash", "chapter", "label", "role"):
            if not isinstance(decision.get(name), str) or not decision[name].strip():
                raise ConceptGraphError(f"coverage: {ident} requires {name}")
        raw_date = decision.get("reviewed_on")
        try:
            reviewed = date.fromisoformat(raw_date) if isinstance(raw_date, str) else raw_date
            if type(reviewed) is not date or reviewed > date.today():
                raise ValueError()
        except (ValueError, TypeError):
            raise ConceptGraphError(f"coverage: {ident} requires a nonfuture ISO reviewed_on") from None
        decision = {**decision, "reviewed_on": reviewed.isoformat()}
        if decision["role"] not in {"primary", "secondary"}:
            raise ConceptGraphError(f"coverage: invalid role {ident}")
        if decision.get("reviewer_kind") not in {"human", "ai"}:
            raise ConceptGraphError(f"coverage: invalid reviewer_kind {ident}")
        targets = decision.get("field_concepts", [])
        if not isinstance(targets, list) or any(not isinstance(t, str) or not re.fullmatch(r"mlsys\.[a-z0-9-]+", t) for t in targets) or len(set(targets)) != len(targets):
            raise ConceptGraphError(f"coverage: invalid field targets {ident}")
        if bool(targets) != (decision["disposition"] in TARGETED):
            raise ConceptGraphError(f"coverage: disposition/targets mismatch {ident}")
        evidence = decision.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ConceptGraphError(f"coverage: decision requires manuscript evidence {ident}")
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"section", "quote"} or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("section", "quote")):
                raise ConceptGraphError(f"coverage: invalid manuscript evidence {ident}")
        if any(len(re.findall(r"\w+", e["quote"])) < 6 for e in evidence):
            raise ConceptGraphError(f"coverage: manuscript quote too short {ident}")
        if len({(e["section"], e["quote"]) for e in evidence}) != len(evidence):
            raise ConceptGraphError(f"coverage: duplicate evidence {ident}")
        decisions[ident] = decision
    aliases = {}
    for concept in field["concepts"]:
        for label in [concept["label"], *concept.get("aliases", [])]:
            aliases.setdefault(_normal(label), set()).add(concept["id"])
    inventory, seen = [], set()
    for chapter in graph["chapters"].values():
        contract = chapter["contract"]
        for role, key in (("primary", "primary_concepts"), ("secondary", "secondary_concepts")):
            for label in contract.get(key, []):
                ident = chapter["id"] + ":" + digest(label)[:16]
                if ident in seen:
                    raise ConceptGraphError(f"coverage: duplicate declaration {ident}: {label}")
                seen.add(ident)
                source_hash, field_hash = decision_hashes(chapter, label, role, field_nodes, {})
                book_ids = {n["id"] for n in graph["nodes"].values() if n["chapter"] == chapter["id"] and n.get("teaches", n.get("concept")) == label}
                passage_suggestions = sorted({link["concept"] for link in field["book_links"] if link["book_concept"] in book_ids})
                item = dict(inventory_id=ident, chapter=chapter["id"], volume=chapter["volume"], label=label, role=role,
                            source=chapter["source"].relative_to(root).as_posix(), source_hash=source_hash, field_hash=field_hash,
                            status="unreviewed", decision=None, stale_reasons=[], suggestions=dict(label_matches=sorted(aliases.get(_normal(label), [])), related_passage_links=passage_suggestions))
                decision = decisions.get(ident)
                if decision:
                    stale = []
                    source_hash, field_hash = decision_hashes(chapter, label, role, field_nodes, decision)
                    item.update(source_hash=source_hash, field_hash=field_hash)
                    if any(decision[k] != item[k] for k in ("chapter", "label", "role")):
                        stale.append("declaration-mismatch")
                    if any(t not in field_nodes for t in decision.get("field_concepts", [])):
                        stale.append("field-target-removed")
                    if decision["source_hash"] != source_hash:
                        stale.append("cited-section-or-declaration-changed")
                    if decision["field_hash"] != field_hash:
                        stale.append("dependent-field-records-changed")
                    for evidence in decision["evidence"]:
                        section = chapter["sections"].get(evidence["section"])
                        if section is None or " ".join(evidence["quote"].split()) not in " ".join(section.split()):
                            stale.append("manuscript-evidence-changed:" + evidence["section"])
                    item.update(status="stale" if stale else "current", decision=decision, stale_reasons=sorted(set(stale)))
                inventory.append(item)
    orphaned = sorted(i for i in set(decisions) - seen if not volumes or i.split("/", 1)[0] in volumes)
    selected = [i for i in inventory if not volumes or i["volume"] in volumes]
    counts = {}
    for volume in sorted({i["volume"] for i in selected}):
        entries = [i for i in selected if i["volume"] == volume]
        counts[volume] = {"declarations": len(entries), "statuses": dict(Counter(i["status"] for i in entries)),
                          "dispositions": dict(Counter(i["decision"]["disposition"] for i in entries if i["status"] == "current")),
                          "dispositions_by_reviewer": {kind: dict(Counter(i["decision"]["disposition"] for i in entries if i["status"] == "current" and i["decision"]["reviewer_kind"] == kind)) for kind in ("human", "ai")},
                          "reviewer_kinds": dict(Counter(i["decision"]["reviewer_kind"] for i in entries if i["status"] == "current")),
                          "unresolved_primary": sum(i["role"] == "primary" and (i["status"] != "current" or i["decision"]["reviewer_kind"] != "human") for i in entries)}
    return dict(schema_version=1, scope="Declared primary and secondary chapter concepts only; manuscript extraction, relation validity, and field completeness are not audited.",
                inventory=selected, by_volume=counts, orphaned_decisions=orphaned,
                gate_passed=bool(selected) and not orphaned and all(i["status"] == "current" and i["decision"]["reviewer_kind"] == "human" for i in selected if i["role"] == "primary"),
                gate_meaning="Gate requirement: every selected primary declaration has a current human disposition against the draft vocabulary, with no orphaned decisions in scope. Secondary declarations are inventoried but not gated. Passing does not certify correctness, teaching depth, or completeness.")


def write_coverage_report(root: Path, volumes: tuple[str, ...] = ()) -> dict:
    report = coverage_report(root, volumes)
    folder = root / "books/_build/concepts"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "-" + "-".join(volumes) if volumes else ""
    (folder / f"coverage-audit{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Chapter concept reconciliation", "", report["scope"], "", "| Volume | Declarations | Unreviewed | Stale | Current human | Current AI | Unresolved primary |", "|---|---:|---:|---:|---:|---:|---:|"]
    for volume, values in report["by_volume"].items():
        statuses = values["statuses"]
        lines.append(f"| {volume} | {values['declarations']} | {statuses.get('unreviewed', 0)} | {statuses.get('stale', 0)} | {values['reviewer_kinds'].get('human', 0)} | {values['reviewer_kinds'].get('ai', 0)} | {values['unresolved_primary']} |")
    lines += ["", "Human reconciliation gate: " + ("passed" if report["gate_passed"] else "not passed"), "", report["gate_meaning"], "", "Orphaned decisions: " + (", ".join(report["orphaned_decisions"]) or "none"), "", "Label matches and related-passage links are suggestions only. See the JSON report for every declaration, evidence, and source hash.", ""]
    (folder / f"coverage-audit{suffix}.md").write_text("\n".join(lines))
    return report
