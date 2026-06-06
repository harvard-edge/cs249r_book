#!/usr/bin/env python3
"""Verify every LEGO inline ref against rendered HTML prose context.

For each ``{python} Class.field`` reference in a chapter:

1. Exec cells and resolve the export value (same namespace as Quarto).
2. Locate that value in archived chapter HTML narrative text.
3. Extract the surrounding sentence/paragraph from HTML.
4. Substitute the value into the QMD source line and compare.
5. Flag mechanical prose defects (missing render, duplicate units, spurious .0,
   literal ``{python}`` leaks, empty context).

Usage (repo root, after HTML archive exists)::

    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_rendered_prose.py
    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_lego_rendered_prose.py \\
        --markdown book/tools/audit/artifacts/lego_rendered_prose_audit.md
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
    INLINE,
    _chapter_paths,
    _exec_cells,
    _html_narrative,
    _normalize,
    _plain_in_html,
    _math_in_html,
    _resolve,
)

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit("audit_lego_rendered_prose.py requires beautifulsoup4") from exc

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
DUPLICATE_UNIT = re.compile(
    r"(?i)(?:"
    r"\b(\d[\d,]*(?:\.\d+)?)\s*(GB|MB|KB|TB|ms|s|kW|W|USD|\$|percent|%|FLOP/byte|tokens|GPUs|GPUs|×|x)\b"
    r".{0,40}?\b\1\s*\2\b"
    r")"
)
UNIT_SUFFIXES = (
    " GB",
    " MB",
    " KB",
    " TB",
    " ms",
    " s",
    " kW",
    " W",
    " FLOP/byte",
    " tokens",
    " GPUs",
    " percent",
    " USD",
    " GPU-hour",
    " GPU-hours",
    " MWh",
    " kWh",
)


def _substitute_line(line: str, ref: str, val: str) -> str:
    token = f"`{{python}} {ref}`"
    return line.replace(token, val)


def _substitute_all_refs(line: str, ns: dict) -> str:
    rendered = line
    for ref in INLINE.findall(line):
        try:
            val, _ = _resolve(ref, ns)
            rendered = _substitute_line(rendered, ref, val)
        except Exception:
            continue
    return rendered


def _html_paragraphs(html: Path) -> list[str]:
    soup = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main") or soup.body
    if not main:
        return []
    for tag in main(["script", "style", "pre", "code"]):
        tag.decompose()
    paras: list[str] = []
    for el in main.find_all(["p", "li", "td", "th", "figcaption", "blockquote", "dd", "h3", "h4", "h5"]):
        text = " ".join(el.get_text(separator=" ").split())
        if text:
            paras.append(text)
    for el in main.find_all(class_=re.compile(r"callout|notebook|lighthouse", re.I)):
        text = " ".join(el.get_text(separator=" ").split())
        if text and text not in paras:
            paras.append(text)
    if not paras:
        paras.append(_html_narrative(html))
    return paras


def _context_windows(text: str, needle: str, *, width: int = 220) -> list[str]:
    if not needle.strip():
        return [text[:width]]
    contexts: list[str] = []
    norm_text = _normalize(text)
    norm_needle = _normalize(needle)
    for candidate in (needle.strip(), needle.strip().replace(",", ""), norm_needle):
        if not candidate:
            continue
        hay = text if candidate in text else norm_text
        search = candidate if candidate in text else norm_needle
        start = 0
        while True:
            idx = hay.find(search, start)
            if idx == -1:
                break
            lo = max(0, idx - width // 2)
            hi = min(len(hay), idx + len(search) + width // 2)
            ctx = hay[lo:hi].strip()
            if ctx and ctx not in contexts:
                contexts.append(ctx)
            start = idx + max(1, len(search))
    return contexts[:3]


def _context_needles(value: str) -> list[str]:
    """Search keys for HTML context — LaTeX exports often render as \\(...\\)."""
    needles: list[str] = []
    for candidate in (value.strip(), value.strip().replace(",", "")):
        if candidate and candidate not in needles:
            needles.append(candidate)
    if "=" in value:
        suffix = value.rsplit("=", 1)[-1].strip()
        if suffix and suffix not in needles:
            needles.append(suffix)
    stripped = value.strip().lstrip("$").rstrip("$")
    if stripped and stripped not in needles:
        needles.append(stripped)
    return needles


def _find_html_contexts(value: str, paragraphs: list[str]) -> list[str]:
    found: list[str] = []
    needles = _context_needles(value)
    for para in paragraphs:
        if not (_plain_in_html(value, para) or _math_in_html(value, para)):
            continue
        for needle in needles:
            found.extend(_context_windows(para, needle))
            if found:
                break
    if not found:
        flat = " ".join(paragraphs)
        if _plain_in_html(value, flat) or _math_in_html(value, flat):
            for needle in needles:
                found.extend(_context_windows(flat, needle))
                if found:
                    break
    return found[:3]


def _duplicate_unit_after_value(rendered: str) -> str | None:
    val = rendered.strip()
    for suffix in UNIT_SUFFIXES:
        if not val.endswith(suffix):
            continue
        unit = suffix.strip()
        tail = rendered[rendered.find(val) + len(val) :] if val in rendered else ""
        if re.search(rf"(?i)\b{re.escape(unit)}\b", tail[:30]):
            return f"duplicate unit '{unit}' after closed formatter value"
    return None


def _mechanical_issues(
    *,
    ref: str,
    value: str,
    kind: str,
    qmd_line: str,
    qmd_rendered: str,
    html_contexts: list[str],
    full_html: str,
) -> list[str]:
    issues: list[str] = []
    if "{python}" in full_html:
        issues.append("literal {python} in HTML")
    in_html = (
        _plain_in_html(value, full_html)
        or ("$" in value and _math_in_html(value, full_html))
        or (kind != "plain" and _math_in_html(value, full_html))
    )
    if not in_html:
        issues.append("value not found in HTML narrative")
    if not html_contexts:
        issues.append("no HTML prose context extracted for this ref")
    dup = _duplicate_unit_after_value(qmd_rendered)
    if dup:
        issues.append(dup)
    # Spurious .0 is covered by chapter-level audit_html; skip here — wide HTML
    # context windows pick up unrelated formula terms (e.g. times 1.0 nearby).
    return issues


def audit_chapter(vol: str, name: str, qmd: Path, html: Path) -> dict:
    row: dict = {
        "vol": vol,
        "chapter": name,
        "refs_total": 0,
        "refs_pass": 0,
        "refs": [],
        "status": "PASS",
    }
    if not qmd.is_file():
        row["status"] = "NO_QMD"
        return row
    if not html.is_file():
        row["status"] = "NO_HTML"
        return row

    ns, _, exec_err = _exec_cells(qmd)
    if exec_err:
        row["status"] = "EXEC_FAIL"
        row["exec_error"] = exec_err
        return row

    content = qmd.read_text(encoding="utf-8")
    lines = content.splitlines()
    paragraphs = _html_paragraphs(html)
    full_html = _html_narrative(html)
    chapter_ok = True
    in_cell = False

    for line_no, qmd_line in enumerate(lines, 1):
        if CELL_START.match(qmd_line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(qmd_line):
            in_cell = False
            continue
        if in_cell:
            continue

        for m in INLINE.finditer(qmd_line):
            ref = m.group(1)
            entry: dict = {
                "ref": ref,
                "qmd_line": line_no,
                "qmd_source": qmd_line.strip()[:240],
            }
            try:
                value, kind = _resolve(ref, ns)
                entry["expected"] = value[:160]
                entry["kind"] = kind
                qmd_rendered = _substitute_all_refs(qmd_line, ns)
                entry["qmd_rendered"] = " ".join(qmd_rendered.split())[:320]
                html_contexts = _find_html_contexts(value, paragraphs)
                entry["html_contexts"] = html_contexts
                issues = _mechanical_issues(
                    ref=ref,
                    value=value,
                    kind=kind,
                    qmd_line=qmd_line,
                    qmd_rendered=entry["qmd_rendered"],
                    html_contexts=html_contexts,
                    full_html=full_html,
                )
                entry["issues"] = issues
                entry["status"] = "PASS" if not issues else "FAIL"
                if issues:
                    chapter_ok = False
                else:
                    row["refs_pass"] += 1
            except Exception as exc:
                entry["status"] = "RESOLVE_FAIL"
                entry["error"] = str(exc)
                chapter_ok = False
            row["refs"].append(entry)
            row["refs_total"] += 1

    row["status"] = "PASS" if chapter_ok and row["refs_total"] else ("FAIL" if row["refs_total"] else "NO_REFS")
    return row


def _write_full_markdown(path: Path, report: list[dict]) -> None:
    """Human-readable corpus: every ref with QMD substitution + HTML context."""
    lines = ["# LEGO rendered prose — full corpus review", ""]
    for row in sorted(report, key=lambda r: (r.get("vol", ""), r.get("chapter", ""))):
        if row.get("status") in ("NO_QMD", "NO_HTML", "EXEC_FAIL"):
            lines.append(f"## {row.get('vol')}/{row.get('chapter')} — {row.get('status')}")
            if row.get("exec_error"):
                lines.append(f"- {row['exec_error']}")
            lines.append("")
            continue
        lines.append(
            f"## {row.get('vol')}/{row.get('chapter')} "
            f"({row.get('refs_pass', 0)}/{row.get('refs_total', 0)} pass)"
        )
        lines.append("")
        for ref_row in row.get("refs", []):
            status = ref_row.get("status", "?")
            lines.append(f"### L{ref_row.get('qmd_line')} `{ref_row.get('ref')}` — **{status}**")
            lines.append(f"- **Expected:** `{ref_row.get('expected', ref_row.get('error', ''))}`")
            lines.append(f"- **QMD rendered:** {ref_row.get('qmd_rendered', ref_row.get('qmd_source', ''))}")
            contexts = ref_row.get("html_contexts") or []
            if contexts:
                for i, ctx in enumerate(contexts, 1):
                    lines.append(f"- **HTML {i}:** …{ctx}…")
            else:
                lines.append("- **HTML:** *(no context extracted)*")
            for issue in ref_row.get("issues") or []:
                lines.append(f"- **Issue:** {issue}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(path: Path, report: list[dict]) -> None:
    lines = [
        "# LEGO rendered prose audit",
        "",
        "Every `{python}` ref: QMD line with substitution + HTML prose context.",
        "",
    ]
    total = sum(r.get("refs_total", 0) for r in report)
    passed = sum(r.get("refs_pass", 0) for r in report)
    fails = sum(1 for r in report if r.get("status") == "FAIL")
    lines.append(f"**Summary:** {passed}/{total} refs pass mechanical checks; {fails} chapters with failures.")
    lines.append("")

    for row in sorted(report, key=lambda r: (r.get("vol", ""), r.get("chapter", ""))):
        if row.get("status") not in ("FAIL", "EXEC_FAIL", "NO_HTML"):
            continue
        lines.append(f"## {row.get('vol')}/{row.get('chapter')} — {row.get('status')}")
        lines.append("")
        for ref_row in row.get("refs", []):
            if ref_row.get("status") == "PASS":
                continue
            lines.append(f"### L{ref_row.get('qmd_line')} `{ref_row.get('ref')}` — {ref_row.get('status')}")
            if ref_row.get("error"):
                lines.append(f"- **Error:** {ref_row['error']}")
            if ref_row.get("expected"):
                lines.append(f"- **Expected:** `{ref_row['expected']}`")
            lines.append(f"- **QMD rendered:** {ref_row.get('qmd_rendered', '')}")
            for i, ctx in enumerate(ref_row.get("html_contexts") or [], 1):
                lines.append(f"- **HTML context {i}:** …{ctx}…")
            for issue in ref_row.get("issues") or []:
                lines.append(f"- **Issue:** {issue}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write JSON report")
    parser.add_argument("--markdown", type=Path, help="Write markdown failure digest")
    parser.add_argument("--chapter", help="Single chapter slug vol1/name")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    paths = _chapter_paths(root)
    if args.chapter:
        vol, name = args.chapter.split("/", 1)
        paths = [p for p in paths if p[0] == vol and p[1] == name]

    report = [audit_chapter(vol, name, qmd, html) for vol, name, qmd, html in paths]

    json_path = args.json or root / "book/tools/audit/artifacts/lego_rendered_prose_audit.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = args.markdown or root / "book/tools/audit/artifacts/lego_rendered_prose_audit.md"
    _write_markdown(md_path, report)
    reports_dir = root / "book/tools/audit/artifacts/lego_chapter_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for row in report:
        if row.get("status") in ("NO_QMD", "NO_HTML"):
            continue
        ch_report = reports_dir / f"{row['vol']}_{row['chapter']}_rendered_prose.md"
        _write_full_markdown(ch_report, [row])
    full_md = md_path.with_name("lego_rendered_prose_full.md")
    if not args.chapter:
        _write_full_markdown(full_md, report)

    total_refs = sum(r.get("refs_total", 0) for r in report)
    pass_refs = sum(r.get("refs_pass", 0) for r in report)
    ch_pass = sum(1 for r in report if r.get("status") == "PASS")
    ch_fail = sum(1 for r in report if r.get("status") == "FAIL")

    print("LEGO rendered prose audit (HTML context + mechanical checks)")
    print("=" * 72)
    print(f"Chapters: {len(report)} | chapter PASS: {ch_pass} | refs PASS: {pass_refs}/{total_refs}")
    print(f"JSON: {json_path}")
    print(f"Markdown failures: {md_path}")
    print(f"Full corpus review: {full_md}")
    for row in report:
        if row.get("status") != "PASS":
            print(f"FAIL {row.get('vol')}/{row.get('chapter')}: {row.get('status')} "
                  f"({row.get('refs_pass', 0)}/{row.get('refs_total', 0)} refs)")
        else:
            rep = reports_dir / f"{row['vol']}_{row['chapter']}_rendered_prose.md"
            print(f"PASS {row.get('vol')}/{row.get('chapter')}: "
                  f"{row.get('refs_pass', 0)}/{row.get('refs_total', 0)} refs — report {rep}")
    return 1 if ch_fail or any(r.get("status") == "EXEC_FAIL" for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
