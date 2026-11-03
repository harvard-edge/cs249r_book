"""Agreement between independent taggers using the ML systems taxonomy.

A taxonomy that things are labeled with is only useful if two people label
the same item the same way. An agreement record (taxonomy/agreement/*.yml)
holds a fixed sample of items and, for each run, every tagger's subject and
concepts per item. This module recomputes agreement from those records, so the
reported numbers can be rerun, and fails when a recorded tag cites a subject or
concept that no longer exists. A run whose rules were revised on the same
sample is marked `held_out: false`; only a fresh sample tests the rules.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .field_graph import load_taxonomy_yaml, load_vocabulary, pct
from .taxonomy_schema import get_class_slots


RECORD_KEYS = set(get_class_slots("AgreementRecordModel"))
RUN_KEYS = set(get_class_slots("AgreementRun"))
TAG_KEYS = set(get_class_slots("AgreementItemTag"))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _index_valid_tags(tags: list, rows: dict) -> dict[int, dict]:
    return {
        t["n"]: t
        for t in tags
        if isinstance(t, dict) and _is_int(t.get("n")) and t.get("subject") in rows
    }


def _kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in left) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def score_run(a: dict, b: dict, rows: dict, home: dict) -> dict:
    """Agreement between two taggers' tags, each keyed by item number."""
    items = sorted(set(a) & set(b))
    subjects = [(a[n]["subject"], b[n]["subject"]) for n in items]
    areas = [(rows[x][1]["code"], rows[y][1]["code"]) for x, y in subjects if x in rows and y in rows]
    both = [n for n in items if a[n].get("concepts") and b[n].get("concepts")]
    both_empty = [n for n in items if not a[n].get("concepts") and not b[n].get("concepts")]
    single_empty = [n for n in items if bool(a[n].get("concepts")) != bool(b[n].get("concepts"))]
    jaccard = [len(set(a[n]["concepts"]) & set(b[n]["concepts"])) / len(set(a[n]["concepts"]) | set(b[n]["concepts"])) for n in both]
    # Rolling (1981): 2c / (A + B). Hooper (1965) is c / (A + B - c), which is the Jaccard index above.
    rolling = [2 * len(set(a[n]["concepts"]) & set(b[n]["concepts"])) / (len(set(a[n]["concepts"])) + len(set(b[n]["concepts"]))) for n in both]
    # All-item set overlap convention: J(empty, empty) = 1.0, J(non-empty, empty) = 0.0.
    all_hooper_sum = sum(jaccard) + float(len(both_empty))
    all_rolling_sum = sum(rolling) + float(len(both_empty))

    def compatible(n):
        x, y = a[n], b[n]
        cx, cy = x.get("concepts", []), y.get("concepts", [])
        return (x["subject"] == y["subject"] or bool(set(cx) & set(cy))
                or any(home.get(c) == y["subject"] for c in cx) or any(home.get(c) == x["subject"] for c in cy))

    cond_hooper_pct = round(100 * sum(jaccard) / len(jaccard), 1) if jaccard else 0.0
    cond_rolling_pct = round(100 * sum(rolling) / len(rolling), 1) if rolling else 0.0
    all_hooper_pct = round(100 * all_hooper_sum / len(items), 1) if items else 0.0
    all_rolling_pct = round(100 * all_rolling_sum / len(items), 1) if items else 0.0

    return {"items": len(items),
            "both_non_empty_items": len(both),
            "single_empty_items": len(single_empty),
            "both_empty_items": len(both_empty),
            "subject_pct": pct(sum(x == y for x, y in subjects), len(subjects)), "subject_kappa": _kappa(subjects),
            "area_pct": pct(sum(x == y for x, y in areas), len(areas)), "area_kappa": _kappa(areas),
            "primary_concept_pct": pct(sum(a[n]["concepts"][0] == b[n]["concepts"][0] for n in both), len(both)),
            "any_shared_concept_pct": pct(sum(j > 0 for j in jaccard), len(both)),
            "same_concept_set_pct": pct(sum(j == 1 for j in jaccard), len(both)),
            "mean_jaccard": round(sum(jaccard) / len(jaccard), 3) if jaccard else 0.0,
            "hooper_consistency_pct": cond_hooper_pct,
            "hooper_conditional_pct": cond_hooper_pct,
            "hooper_all_items_pct": all_hooper_pct,
            "rolling_consistency_pct": cond_rolling_pct,
            "rolling_conditional_pct": cond_rolling_pct,
            "rolling_all_items_pct": all_rolling_pct,
            "compatible_pct": pct(sum(compatible(n) for n in items), len(items))}


def _check(record: dict, concepts: set, rows: dict, name: str) -> list[str]:
    errors = [f"{name}: requires {key}" for key in ("id", "title", "basis") if not str(record.get(key, "")).strip()]
    if set(record) - RECORD_KEYS:
        errors.append(f"{name}: unknown record keys: {', '.join(sorted(set(record) - RECORD_KEYS))}")
    items = record.get("items", [])
    if not isinstance(items, list):
        return errors + [f"{name}: items must be a list"]
    numbers = {item["n"] for item in items if isinstance(item, dict) and _is_int(item.get("n"))}
    if len(numbers) != len(items) or not numbers:
        errors.append(f"{name}: items must have distinct sample numbers")
    runs_list = record.get("runs", [])
    if not isinstance(runs_list, list):
        return errors + [f"{name}: runs must be a list"]
    for run in runs_list:
        if not isinstance(run, dict):
            errors.append(f"{name}: run must be a mapping")
            continue
        if set(run) - RUN_KEYS:
            errors.append(f"{name} run {run.get('id')}: unknown run keys: {', '.join(sorted(set(run) - RUN_KEYS))}")
        if not isinstance(run.get("held_out"), bool):
            errors.append(f"{name} run {run.get('id')}: requires held_out true or false")
        taggers = run.get("taggers", {})
        if not isinstance(taggers, dict) or len(taggers) != 2:
            errors.append(f"{name} run {run.get('id')}: requires exactly two taggers")
            continue
        for tagger, tags in taggers.items():
            where = f"{name} run {run.get('id')} tagger {tagger}"
            if not isinstance(tags, list):
                errors.append(f"{where}: tags must be a list")
                continue
            valid_tags = [
                t for t in tags
                if isinstance(t, dict) and _is_int(t.get("n")) and isinstance(t.get("subject"), str)
            ]
            if len(valid_tags) != len(tags):
                errors.append(f"{where}: every tag requires integer n and string subject")
                continue
            if len(valid_tags) != len(numbers) or {t["n"] for t in valid_tags} != numbers:
                errors.append(f"{where}: does not tag every sample item exactly once")
            for t in valid_tags:
                if set(t) - TAG_KEYS:
                    errors.append(f"{where} item {t.get('n')}: unknown tag keys: {', '.join(sorted(set(t) - TAG_KEYS))}")
                if t["subject"] not in rows:
                    errors.append(f"{where} item {t['n']}: unknown subject {t['subject']}")
                cs = t.get("concepts", [])
                if len(set(cs)) != len(cs):
                    errors.append(f"{where} item {t['n']}: duplicate concept tags")
                gone = [c for c in cs if c not in concepts]
                if gone:
                    errors.append(f"{where} item {t['n']}: cites concepts that no longer exist: {', '.join(gone)}")
    return errors


def agreement_report(root: Path) -> dict:
    """Validate all tagging-agreement records under `taxonomy/agreement/` and compute inter-tagger scores."""
    from .concept_graph import ConceptGraphError
    from .taxonomy_schema import validate_linkml
    concepts, rows, home = load_vocabulary(root, "agreement")
    records, errors = [], []
    for path in sorted((root / "taxonomy/agreement").glob("*.yml")):
        record = load_taxonomy_yaml(path)
        errors += _check(record, concepts, rows, path.name)
        try:
            validate_linkml(record, "AgreementRecordModel", root, path.name)
        except ConceptGraphError as exc:
            errors.append(str(exc))
        runs = []
        for run in record.get("runs", []) if isinstance(record.get("runs"), list) else []:
            if not isinstance(run, dict):
                continue
            taggers = run.get("taggers")
            if not isinstance(taggers, dict) or len(taggers) != 2:
                continue
            (name_a, tags_a), (name_b, tags_b) = list(taggers.items())
            if not isinstance(tags_a, list) or not isinstance(tags_b, list):
                continue
            runs.append({"id": run.get("id", "?"), "held_out": run.get("held_out"), "note": run.get("note", ""), "taggers": [name_a, name_b],
                         "scores": score_run(_index_valid_tags(tags_a, rows), _index_valid_tags(tags_b, rows), rows, home)})
        records.append({"id": record.get("id", "?"), "title": record.get("title", ""), "basis": record.get("basis", ""), "file": path.name, "runs": runs})
    return {"records": records, "errors": errors}


def write_report(root: Path) -> dict:
    """Compute the tagging-agreement report and write `books/_build/concepts/taxonomy-agreement.json`."""
    report = agreement_report(root)
    out = root / "books/_build/concepts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy-agreement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
