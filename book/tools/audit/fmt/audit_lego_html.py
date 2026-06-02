#!/usr/bin/env python3
"""Verify archived chapter HTML contains values from inline-python LEGO exports.

For each QMD chapter with archived HTML under ``html-audit/<vol>/``:

1. Exec python cells (shared namespace).
2. Group ``{python} Class.attr`` refs by class (LEGO focal point).
3. Resolve each export from Python.
4. Check rendered HTML:
   - ``*_str`` plain exports: literal substring in narrative HTML text.
   - ``*_math`` / ``*_eq`` / ``MarkdownStr``: key numeric tokens appear in HTML
     (math renders as LaTeX spans, not plain strings).

Usage (repo root)::

    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_html.py
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_html.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit("audit_lego_html.py requires beautifulsoup4") from exc

from cell_exec import exec_cell_code, make_exec_namespace, setup_headless_matplotlib

setup_headless_matplotlib()

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
INLINE = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
CLASS = re.compile(r"^class\s+(\w+)", re.M)
LEGO_MARK = re.compile(r"#\s*[│┌].*LEGO|#\s*\│ Exports:")
NUM_TOKEN = re.compile(r"\d[\d,]*\.?\d*")

CHAPTER_LIST = {
    "vol1": [
        "introduction", "ml_systems", "ml_workflow", "data_engineering", "nn_computation",
        "nn_architectures", "frameworks", "training", "data_selection", "model_compression",
        "hw_acceleration", "benchmarking", "model_serving", "ml_ops", "responsible_engr",
        "conclusion", "appendix_algorithm", "appendix_assumptions",
        "appendix_data", "appendix_machine",
    ],
    "vol2": [
        "introduction", "compute_infrastructure", "network_fabrics", "data_storage",
        "distributed_training", "collective_communication", "fault_tolerance",
        "fleet_orchestration", "performance_engineering", "inference", "edge_intelligence",
        "ops_scale", "security_privacy", "robust_ai", "sustainable_ai", "responsible_ai",
        "conclusion", "appendix_dam", "appendix_fleet", "appendix_communication",
        "appendix_reliability", "appendix_inference", "appendix_c3",
        "appendix_assumptions",
    ],
}


def _chapter_paths(root: Path) -> list[tuple[str, str, Path, Path]]:
    out: list[tuple[str, str, Path, Path]] = []
    for vol, names in CHAPTER_LIST.items():
        for name in names:
            if name.startswith("appendix_"):
                qmd = root / f"book/quarto/contents/{vol}/backmatter/{name}.qmd"
            else:
                qmd = root / f"book/quarto/contents/{vol}/{name}/{name}.qmd"
            html = root / f"book/quarto/_build/html-audit/{vol}/{name}.html"
            if not html.is_file():
                if name.startswith("appendix_"):
                    html = (
                        root
                        / f"book/quarto/_build/html-{vol}/contents/{vol}/backmatter/{name}.html"
                    )
                else:
                    html = (
                        root
                        / f"book/quarto/_build/html-{vol}/contents/{vol}/{name}/{name}.html"
                    )
            out.append((vol, name, qmd, html))
    return out


def _exec_cells(qmd: Path) -> tuple[dict, set[str], str | None]:
    lines = qmd.read_text(encoding="utf-8").splitlines()
    ns = make_exec_namespace()
    lego: set[str] = set()
    in_cell = False
    buf: list[str] = []
    for line in lines:
        if CELL_START.match(line):
            in_cell = True
            buf = []
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            code = "\n".join(buf)
            is_lego = bool(LEGO_MARK.search(code)) or "Exports:" in code
            m = CLASS.search(code)
            cls = m.group(1) if m else None
            try:
                exec_cell_code(code, ns)
            except Exception as exc:
                return ns, lego, str(exc)
            if cls and is_lego:
                lego.add(cls)
            continue
        if in_cell:
            buf.append(line)
    return ns, lego, None


def _resolve(ref: str, ns: dict) -> tuple[str, str]:
    parts = ref.split(".")
    obj = ns[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    kind = "math" if ref.endswith(("_math", "_eq", "_frac", "_eqn_str")) else "plain"
    return str(obj), kind


def _html_narrative(html: Path) -> str:
    soup = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main") or soup.body
    for tag in main(["script", "style", "pre", "code"]):
        tag.decompose()
    return " ".join(main.get_text(separator=" ").split())


def _normalize(s: str) -> str:
    s = (
        s.replace("\\$", "$")
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "")
        .replace("~", "")
        .replace("--", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00d7", "x")
        .replace("×", "x")
        .replace("\u2248", "")
        .replace("≈", "")
        .replace("÷", "/")
        .replace("\u2082", "2")  # CO₂ subscript
        .replace("₂", "2")
        .lower()
    )
    s = re.sub(r"co\s*2", "co2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _plain_in_html(value: str, ht: str) -> bool:
    if not value.strip():
        return True
    candidates = [
        value.strip(),
        value.strip().replace(",", ""),
        value.strip().lstrip("$").rstrip("$"),
        _normalize(value),
    ]
    norm_ht = _normalize(ht)
    return any(c and (c in ht or _normalize(c) in norm_ht) for c in candidates)


def _math_in_html(value: str, ht: str) -> bool:
    """Math exports: require key LaTeX fragments or numeric literals in HTML."""
    norm_ht = _normalize(ht)
    # Pure-formula exports (no standalone numbers): match distinctive substrings.
    stripped = value.strip().lstrip("$").rstrip("$")
    if stripped and not NUM_TOKEN.findall(stripped.replace(",", "")):
        for frag in re.split(r"[=\\{}]+", stripped):
            frag = frag.strip().lower()
            if len(frag) >= 4 and frag in norm_ht:
                return True
    signed = re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    nums = NUM_TOKEN.findall(value.replace(",", ""))
    sig = list(dict.fromkeys(signed + nums))
    sig = [n for n in sig if n and (n.lstrip("-") not in {"0", "1", "2"} or len(n.lstrip("-")) > 1 or n.startswith("-"))]
    if not sig:
        chunk = re.sub(r"[^a-zA-Z0-9.%+-]", "", _normalize(value))
        return bool(chunk and chunk[:12] in norm_ht)
    return all(n.replace(",", "") in norm_ht or n in ht for n in sig)


def audit_chapter(vol: str, name: str, qmd: Path, html: Path) -> dict:
    row: dict = {"vol": vol, "chapter": name, "qmd": str(qmd), "html": str(html)}
    if not qmd.is_file():
        row["status"] = "NO_QMD"
        return row
    if not html.is_file():
        row["status"] = "NO_HTML"
        return row

    ns, lego, err = _exec_cells(qmd)
    if err:
        row["status"] = "EXEC_FAIL"
        row["error"] = err
        return row

    ht = _html_narrative(html)
    refs_by_class: dict[str, set[str]] = defaultdict(set)
    for m in INLINE.finditer(qmd.read_text(encoding="utf-8")):
        refs_by_class[m.group(1).split(".")[0]].add(m.group(1))

    classes = sorted(set(lego) | set(refs_by_class.keys()))
    blocks: list[dict] = []
    chapter_ok = True

    for cls in classes:
        refs = sorted(refs_by_class.get(cls, []))
        block = {
            "class": cls,
            "lego_marked": cls in lego,
            "refs_in_prose": len(refs),
            "refs": [],
            "status": "NO_REFS" if not refs else "PASS",
        }
        if not refs:
            blocks.append(block)
            continue

        block_ok = True
        for ref in refs:
            entry = {"ref": ref}
            try:
                val, kind = _resolve(ref, ns)
            except Exception as exc:
                entry.update({"status": "RESOLVE_FAIL", "error": str(exc)})
                block_ok = False
                block["refs"].append(entry)
                continue

            if kind == "plain":
                found = _plain_in_html(val, ht) or ("$" in val and _math_in_html(val, ht))
            else:
                found = _math_in_html(val, ht)
            entry.update({
                "kind": kind,
                "expected_preview": val[:100],
                "in_html": found,
                "status": "PASS" if found else "FAIL",
            })
            if not found:
                block_ok = False
            block["refs"].append(entry)

        block["status"] = "PASS" if block_ok else "FAIL"
        if not block_ok:
            chapter_ok = False
        blocks.append(block)

    row["status"] = "PASS" if chapter_ok else "FAIL"
    row["lego_blocks"] = len([b for b in blocks if b["refs_in_prose"] > 0])
    row["lego_blocks_pass"] = len([b for b in blocks if b["status"] == "PASS" and b["refs_in_prose"] > 0])
    row["blocks"] = blocks
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument("--report", type=Path, help="Write JSON report path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    report = [audit_chapter(vol, name, qmd, html) for vol, name, qmd, html in _chapter_paths(root)]

    out_path = args.report or root / "book/quarto/_build/html-audit/lego_html_verify_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    total = len(report)
    ch_pass = sum(1 for r in report if r["status"] == "PASS")
    print("LEGO × HTML verification report")
    print("=" * 72)
    print(f"Chapters: {total} | PASS: {ch_pass} | FAIL: {total - ch_pass}")
    print(f"Full report: {out_path}\n")

    for r in report:
        if r["status"] != "PASS":
            print(f"FAIL {r['vol']}/{r['chapter']}: {r.get('error', r['status'])}")
            for b in r.get("blocks", []):
                if b["status"] == "FAIL":
                    miss = [x["ref"] for x in b["refs"] if x.get("status") == "FAIL"]
                    resolve = [x["ref"] for x in b["refs"] if x.get("status") == "RESOLVE_FAIL"]
                    if miss:
                        print(f"  {b['class']}: not in HTML ({len(miss)}) -> {miss[:4]}")
                    if resolve:
                        print(f"  {b['class']}: resolve fail ({len(resolve)}) -> {resolve[:4]}")
            continue
        n = r.get("lego_blocks", 0)
        print(f"PASS {r['vol']}/{r['chapter']}: {n} LEGO blocks, all refs found in HTML")

    return 0 if ch_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
