"""Agreement between independent taggers using the ML systems taxonomy.

A taxonomy that things are labeled with is only useful if two people label
the same item the same way. An agreement record (shared/concepts/agreement/*.yml)
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

import yaml


def _vocabulary(root: Path):
    field = yaml.safe_load((root / "shared/concepts/ml-systems-field.yml").read_text())
    framework = yaml.safe_load((root / "shared/concepts/ml-systems-framework.yml").read_text())
    rows = {row["id"]: (row, area) for area in framework["areas"] for row in area["rows"]}
    home = {c: rid for rid, (row, _) in rows.items() for c in row.get("concepts", [])}
    return {c["id"] for c in field["concepts"]}, rows, home


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def _kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
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
    jaccard = [len(set(a[n]["concepts"]) & set(b[n]["concepts"])) / len(set(a[n]["concepts"]) | set(b[n]["concepts"])) for n in both]

    def compatible(n):
        x, y = a[n], b[n]
        cx, cy = x.get("concepts", []), y.get("concepts", [])
        return (x["subject"] == y["subject"] or bool(set(cx) & set(cy))
                or any(home.get(c) == y["subject"] for c in cx) or any(home.get(c) == x["subject"] for c in cy))

    return {"items": len(items),
            "subject_pct": _pct(sum(x == y for x, y in subjects), len(subjects)), "subject_kappa": _kappa(subjects),
            "area_pct": _pct(sum(x == y for x, y in areas), len(areas)), "area_kappa": _kappa(areas),
            "primary_concept_pct": _pct(sum(a[n]["concepts"][0] == b[n]["concepts"][0] for n in both), len(both)),
            "any_shared_concept_pct": _pct(sum(j > 0 for j in jaccard), len(both)),
            "same_concept_set_pct": _pct(sum(j == 1 for j in jaccard), len(both)),
            "mean_jaccard": round(sum(jaccard) / len(jaccard), 3) if jaccard else 0.0,
            "compatible_pct": _pct(sum(compatible(n) for n in items), len(items))}


def _check(record: dict, concepts: set, rows: dict, name: str) -> list[str]:
    errors = [f"{name}: requires {key}" for key in ("id", "title", "basis") if not str(record.get(key, "")).strip()]
    numbers = {item["n"] for item in record.get("items", [])}
    for run in record.get("runs", []):
        taggers = run.get("taggers", {})
        if len(taggers) != 2:
            errors.append(f"{name} run {run.get('id')}: requires exactly two taggers")
        for tagger, tags in taggers.items():
            where = f"{name} run {run.get('id')} tagger {tagger}"
            if {t["n"] for t in tags} != numbers:
                errors.append(f"{where}: does not tag every sample item exactly once")
            for t in tags:
                if t["subject"] not in rows:
                    errors.append(f"{where} item {t['n']}: unknown subject {t['subject']}")
                gone = [c for c in t.get("concepts", []) if c not in concepts]
                if gone:
                    errors.append(f"{where} item {t['n']}: cites concepts that no longer exist: {', '.join(gone)}")
    return errors


def agreement_report(root: Path) -> dict:
    concepts, rows, home = _vocabulary(root)
    records, errors = [], []
    for path in sorted((root / "shared/concepts/agreement").glob("*.yml")):
        record = yaml.safe_load(path.read_text())
        errors += _check(record, concepts, rows, path.name)
        runs = []
        for run in record.get("runs", []):
            (name_a, tags_a), (name_b, tags_b) = list(run["taggers"].items())[:2]
            index = lambda tags: {t["n"]: t for t in tags if t["subject"] in rows}
            runs.append({"id": run["id"], "held_out": run.get("held_out"), "note": run.get("note", ""), "taggers": [name_a, name_b],
                         "scores": score_run(index(tags_a), index(tags_b), rows, home)})
        records.append({"id": record["id"], "title": record["title"], "basis": record["basis"], "file": path.name, "runs": runs})
    return {"records": records, "errors": errors}


def write_report(root: Path) -> dict:
    report = agreement_report(root)
    out = root / "books/_build/concepts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy-agreement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
