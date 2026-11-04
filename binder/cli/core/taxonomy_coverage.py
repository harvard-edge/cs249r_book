"""Coverage of the ML systems taxonomy against recorded external checklists.

A checklist (taxonomy/coverage/*.yml) lists topics from a bounded,
named source, such as course syllabi or a conference's accepted papers, with
an authored judgment for each: covered by specific concepts, partially covered
by a subject, missing, or out of scope. The judgment is editorial; this module
keeps it honest. It fails when a judgment cites a concept or subject that no
longer exists, and it computes coverage in code so every reported number can
be rerun after the taxonomy changes. A checklist that informed the taxonomy's
design is marked `held_out: false`; only held-out checklists test it.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .field_graph import load_taxonomy_yaml, load_vocabulary, pct
from .taxonomy_schema import get_class_slots, get_enum_values


STATUSES = get_enum_values("ChecklistStatusEnum")
CHECKLIST_KEYS = set(get_class_slots("CoverageChecklistModel"))
TOPIC_KEYS = set(get_class_slots("CoverageChecklistTopic"))


def _check(checklist: dict, concepts: set, subjects: dict, homes: dict, name: str) -> list[str]:
    errors = []
    if set(checklist) - CHECKLIST_KEYS:
        errors.append(f"{name}: unknown checklist keys: {', '.join(sorted(set(checklist) - CHECKLIST_KEYS))}")
    for key in ("id", "title", "basis"):
        if not str(checklist.get(key, "")).strip():
            errors.append(f"{name}: requires {key}")
    if not isinstance(checklist.get("held_out"), bool):
        errors.append(f"{name}: requires held_out true or false")
    topics = checklist.get("topics")
    if not isinstance(topics, list) or not topics:
        return errors + [f"{name}: requires a nonempty topics list"]
    for index, item in enumerate(topics, 1):
        where = f"{name} topic {index} ({item.get('topic', '?')})"
        if set(item) - TOPIC_KEYS:
            errors.append(f"{where}: unknown topic keys: {', '.join(sorted(set(item) - TOPIC_KEYS))}")
        if not isinstance(item.get("topic"), str) or not item["topic"].strip():
            errors.append(f"{where}: requires topic")
        status = item.get("status")
        if status not in STATUSES:
            errors.append(f"{where}: unknown status {status}")
        unknown = [c for c in item.get("concepts", []) if c not in concepts]
        if unknown:
            errors.append(f"{where}: cites concepts that no longer exist: {', '.join(unknown)}")
        if item.get("subject") and item["subject"] not in subjects:
            errors.append(f"{where}: cites unknown subject {item['subject']}")
        if status == "covered" and not item.get("concepts"):
            errors.append(f"{where}: covered requires at least one concept")
        # 2026-10-27: 13 covered topics kept a stale subject after concepts moved
        # or were added in another row. When a covered topic names a subject, it
        # must match the home of at least one cited concept.
        if status == "covered" and item.get("subject") in subjects:
            concept_homes = {homes[c] for c in item.get("concepts", []) if c in homes}
            if concept_homes and item["subject"] not in concept_homes:
                errors.append(f"{where}: subject {item['subject']} disagrees with concept homes {sorted(concept_homes)}")
        if status == "partial" and not item.get("subject"):
            errors.append(f"{where}: partial requires a subject")
        if status in ("missing", "out-of-scope") and item.get("concepts"):
            errors.append(f"{where}: {status} must not cite concepts")
    return errors


def _summary(topics: list[dict]) -> dict:
    counts = Counter(t["status"] for t in topics)
    weights = Counter()
    for t in topics:
        weights[t["status"]] += max(1, len(t.get("sources", [])))
    in_scope = len(topics) - counts["out-of-scope"]
    weight_in_scope = sum(weights.values()) - weights["out-of-scope"]

    return {"topics": len(topics), "in_scope": in_scope, "counts": {s: counts[s] for s in STATUSES},
            "covered_pct": pct(counts["covered"], in_scope),
            "homed_pct": pct(counts["covered"] + counts["partial"], in_scope),
            "weighted_covered_pct": pct(weights["covered"], weight_in_scope),
            "weighted_homed_pct": pct(weights["covered"] + weights["partial"], weight_in_scope)}


def coverage_report(root: Path) -> dict:
    """Validate all external coverage checklists under `taxonomy/coverage/` and compute summary metrics."""
    from .concept_graph import ConceptGraphError
    from .taxonomy_schema import validate_linkml
    folder = root / "taxonomy/coverage"
    concepts, subjects, homes = load_vocabulary(root, "coverage")
    checklists, errors = [], []
    for path in sorted(folder.glob("*.yml")):
        checklist = load_taxonomy_yaml(path)
        errors += _check(checklist, concepts, subjects, homes, path.name)
        try:
            validate_linkml(checklist, "CoverageChecklistModel", root, path.name)
        except ConceptGraphError as exc:
            errors.append(str(exc))
        if isinstance(checklist.get("topics"), list):
            checklists.append({"id": checklist.get("id"), "title": checklist.get("title"), "basis": checklist.get("basis"),
                               "held_out": checklist.get("held_out"), "file": path.name,
                               "summary": _summary([t for t in checklist["topics"] if t.get("status") in STATUSES]),
                               "open": [t for t in checklist["topics"] if t.get("status") in ("partial", "missing")]})
    return {"checklists": checklists, "errors": errors}


def write_report(root: Path) -> dict:
    """Compute the external checklist coverage report and write markdown and JSON summaries to `books/_build/concepts/`."""
    report = coverage_report(root)
    out = root / "books/_build/concepts"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Taxonomy coverage against recorded checklists", "",
             "Judgments are editorial and recorded in taxonomy/coverage/. Numbers are computed by `binder concepts coverage`.", ""]
    for c in report["checklists"]:
        s = c["summary"]
        lines += [f"## {c['title']}", "", f"Basis: {c['basis']}", "",
                  f"Held out from design: {'yes' if c['held_out'] else 'no (this checklist informed the taxonomy, so it is not an independent test)'}", "",
                  f"- In-scope topics: {s['in_scope']} of {s['topics']}",
                  f"- Covered by a concept: {s['counts']['covered']} ({s['covered_pct']}%), weighted by sources {s['weighted_covered_pct']}%",
                  f"- Covered or homed in a subject: {s['counts']['covered'] + s['counts']['partial']} ({s['homed_pct']}%), weighted {s['weighted_homed_pct']}%",
                  f"- Partial: {s['counts']['partial']} · Missing: {s['counts']['missing']} · Out of scope: {s['counts']['out-of-scope']}", ""]
        if c["open"]:
            lines += ["Open topics:", ""] + [f"- {t['status']}: {t['topic']}" + (f" ({t['subject']})" if t.get("subject") else "") for t in c["open"]] + [""]
    if report["errors"]:
        lines += ["## Invalid judgments", ""] + [f"- {e}" for e in report["errors"]]
    (out / "taxonomy-coverage.md").write_text("\n".join(lines) + "\n")
    (out / "taxonomy-coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
