#!/usr/bin/env python3
"""Verify every LEGO cell in archived chapters: exec, outputs, HTML refs.

For each ``{python}`` cell marked by a LEGO header or output section:

1. Exec all chapter cells in order (shared namespace — matches Quarto render).
2. List every ``*_str`` / ``*_math`` / ``*_eq`` / ``*_frac`` output on the class.
3. Confirm each output resolves after exec.
4. For each ``{python} Class.output`` ref in prose, confirm the rendered value
   appears in archived HTML (same rules as ``audit_lego_html.py``).

Usage (repo root)::

    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_cells.py
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_cells.py \\
        --report book/tools/audit/artifacts/lego_cells_verify_report.json
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_cells.py \\
        --chapter vol2/network_fabrics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_lego_html import (  # noqa: E402
    CHAPTER_LIST,
    CLASS,
    INLINE,
    LEGO_MARK,
    _chapter_paths,
    _exec_cells,
    _html_narrative,
    _math_in_html,
    _plain_in_html,
    _resolve,
)
from cell_exec import exec_cell_code, make_exec_namespace  # noqa: E402

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
EXPORT = re.compile(r"^\s+(\w+(?:_str|_math|_eq|_frac))\s*=", re.M)
GOAL = re.compile(r"#\s*[│├].*Goal:\s*(.+)", re.I)


def _classes_in_code(code: str) -> list[str]:
    return CLASS.findall(code)


def _exports_for_class(code: str, cls: str) -> list[str]:
    """Exports declared on ``cls`` body only (stop at next top-level class)."""
    m = re.search(rf"^class\s+{re.escape(cls)}\s*[:\(]", code, re.M)
    if not m:
        return []
    rest = code[m.end() :]
    nxt = re.search(r"^class\s+\w+", rest, re.M)
    block = rest[: nxt.start()] if nxt else rest
    return EXPORT.findall(block)


def _parse_lego_cells(qmd: Path) -> list[dict]:
    lines = qmd.read_text(encoding="utf-8").splitlines()
    cells: list[dict] = []
    in_cell = False
    buf: list[str] = []
    start_line = 0
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            buf = []
            start_line = i
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            code = "\n".join(buf)
            is_lego = bool(LEGO_MARK.search(code))
            if not is_lego:
                continue
            goal_m = GOAL.search(code)
            goal = goal_m.group(1).strip() if goal_m else ""
            for cls in _classes_in_code(code):
                cells.append(
                    {
                        "cell_line": start_line,
                        "class": cls,
                        "is_lego": True,
                        "code": code,
                        "exports": _exports_for_class(code, cls),
                        "goal": goal,
                    }
                )
            continue
        if in_cell:
            buf.append(line)
    return cells


def _exec_with_cell_errors(
    qmd: Path,
) -> tuple[dict, set[str], dict[str, str], list[tuple[str, str]]]:
    """Exec cells; attribute exec failures to the cell line that failed.

    Returns ``(namespace, lego_classes, errors, cell_warnings)`` where
    ``cell_warnings`` is a list of ``(class_or_cell_key, warning_message)``
    tuples captured during execution.
    """
    import warnings as _warnings

    lines = qmd.read_text(encoding="utf-8").splitlines()
    ns = make_exec_namespace()
    lego: set[str] = set()
    errors: dict[str, str] = {}
    cell_warnings: list[tuple[str, str]] = []
    in_cell = False
    buf: list[str] = []
    start_line = 0

    # Warning categories worth flagging during cell exec
    _WARN_KEYWORDS = ("Precision", "not found", "deprecated", "Deprecated")

    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            buf = []
            start_line = i
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            code = "\n".join(buf)
            is_lego = bool(LEGO_MARK.search(code))
            m = CLASS.search(code)
            cls = m.group(1) if m else None
            key = cls or f"cell@{start_line}"
            try:
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter("always")
                    exec_cell_code(code, ns)
                    for w in caught:
                        msg = str(w.message)
                        if w.category is UserWarning and any(
                            kw in msg for kw in _WARN_KEYWORDS
                        ):
                            cell_warnings.append((key, msg))
            except Exception as exc:
                errors[key] = str(exc)
                return ns, lego, errors, cell_warnings
            if is_lego:
                lego.update(_classes_in_code(code))
            continue
        if in_cell:
            buf.append(line)
    return ns, lego, errors, cell_warnings


def audit_chapter(vol: str, name: str, qmd: Path, html: Path) -> dict:
    row: dict = {
        "vol": vol,
        "chapter": name,
        "qmd": str(qmd),
        "html": str(html),
        "cells": [],
    }
    if not qmd.is_file():
        row["status"] = "NO_QMD"
        return row
    if not html.is_file():
        row["status"] = "NO_HTML"
        return row

    parsed = _parse_lego_cells(qmd)
    ns, lego_classes, exec_errors, cell_warnings = _exec_with_cell_errors(qmd)
    if cell_warnings:
        row["cell_warnings"] = [
            {"class": ckey, "message": wmsg} for ckey, wmsg in cell_warnings
        ]
    if exec_errors:
        row["status"] = "EXEC_FAIL"
        row["exec_error"] = exec_errors
        row["cells_total"] = len(parsed)
        return row

    ht = _html_narrative(html)
    content = qmd.read_text(encoding="utf-8")
    refs_by_class: dict[str, set[str]] = defaultdict(set)
    for m in INLINE.finditer(content):
        refs_by_class[m.group(1).split(".")[0]].add(m.group(1))

    chapter_ok = True
    for cell in parsed:
        cls = cell["class"]
        record: dict = {
            "cell_line": cell["cell_line"],
            "class": cls,
            "goal": cell["goal"],
            "exports_declared": cell["exports"],
            "exports": [],
            "refs": [],
            "status": "PASS",
        }

        export_ok = True
        for attr in cell["exports"]:
            ref = f"{cls}.{attr}"
            entry = {"export": attr, "ref": ref}
            try:
                val, kind = _resolve(ref, ns)
                entry["resolved_preview"] = val[:120]
                entry["status"] = "PASS"
            except Exception as exc:
                entry["status"] = "RESOLVE_FAIL"
                entry["error"] = str(exc)
                export_ok = False
            record["exports"].append(entry)

        refs = sorted(refs_by_class.get(cls, []))
        ref_ok = True
        for ref in refs:
            entry = {"ref": ref}
            try:
                val, kind = _resolve(ref, ns)
                if kind == "plain":
                    found = _plain_in_html(val, ht) or (
                        "$" in val and _math_in_html(val, ht)
                    )
                else:
                    found = _math_in_html(val, ht)
                entry.update(
                    {
                        "kind": kind,
                        "expected_preview": val[:120],
                        "in_html": found,
                        "status": "PASS" if found else "FAIL",
                    }
                )
                if not found:
                    ref_ok = False
            except Exception as exc:
                entry.update({"status": "RESOLVE_FAIL", "error": str(exc)})
                ref_ok = False
            record["refs"].append(entry)

        if not export_ok or not ref_ok:
            record["status"] = "FAIL"
            chapter_ok = False
        elif not refs:
            record["status"] = "NO_PROSE_REFS"
            chapter_ok = False
        row["cells"].append(record)

    row["cells_total"] = len(parsed)
    row["cells_pass"] = sum(1 for c in row["cells"] if c["status"] == "PASS")
    row["status"] = "PASS" if chapter_ok and parsed else ("FAIL" if parsed else "NO_LEGO")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument("--report", type=Path, help="Write JSON report path")
    parser.add_argument(
        "--chapter",
        help="Single chapter slug, e.g. vol2/network_fabrics",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    paths = _chapter_paths(root)
    if args.chapter:
        vol, name = args.chapter.split("/", 1)
        paths = [p for p in paths if p[0] == vol and p[1] == name]
        if not paths:
            print(f"Unknown chapter: {args.chapter}", file=sys.stderr)
            return 2

    report = [audit_chapter(vol, name, qmd, html) for vol, name, qmd, html in paths]

    out_path = args.report or (
        root / "book/tools/audit/artifacts/lego_cells_verify_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_cells = sum(r.get("cells_total", 0) for r in report)
    pass_cells = sum(r.get("cells_pass", 0) for r in report)
    ch_pass = sum(1 for r in report if r["status"] == "PASS")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if ch_pass == len(report) else 1

    print("LEGO cell verification (exec + outputs + HTML refs)")
    print("=" * 72)
    print(
        f"Chapters: {len(report)} | chapter PASS: {ch_pass} | "
        f"cells PASS: {pass_cells}/{total_cells}"
    )
    print(f"Full report: {out_path}\n")

    failed = False
    for r in report:
        # Surface any cell-execution warnings (e.g. Precision, deprecated)
        for cw in r.get("cell_warnings", []):
            print(
                f"WARN {r['vol']}/{r['chapter']}: "
                f"{cw['class']} emitted UserWarning: {cw['message']}"
            )

        if r["status"] == "PASS":
            print(
                f"PASS {r['vol']}/{r['chapter']}: "
                f"{r['cells_pass']}/{r['cells_total']} LEGO cells"
            )
            continue
        failed = True
        print(f"FAIL {r['vol']}/{r['chapter']}: {r.get('exec_error', r['status'])}")
        for cell in r.get("cells", []):
            if cell["status"] != "PASS":
                bad_exports = [
                    e["export"]
                    for e in cell.get("exports", [])
                    if e.get("status") != "PASS"
                ]
                bad_refs = [
                    e["ref"]
                    for e in cell.get("refs", [])
                    if e.get("status") not in ("PASS",)
                ]
                print(
                    f"  L{cell['cell_line']} {cell['class']}: {cell['status']} "
                    f"outputs={bad_exports[:3]} refs={bad_refs[:3]}"
                )

    return 1 if failed or ch_pass != len(report) else 0


if __name__ == "__main__":
    sys.exit(main())
