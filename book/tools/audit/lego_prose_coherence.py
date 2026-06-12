#!/usr/bin/env python3
"""LLM coherence review for LEGO inline-python prose.

Extracts prose lines that reference ``{python} Class.field`` outputs (with
values substituted from cell exec), groups them by LEGO class, and asks an LLM
whether each snippet reads coherently in context.

Usage (repo root)::

    PYTHONPATH=mlsysim python3 book/tools/audit/lego_prose_coherence.py --chapter vol2/network_fabrics
    PYTHONPATH=mlsysim python3 book/tools/audit/lego_prose_coherence.py --all
    PYTHONPATH=mlsysim python3 book/tools/audit/lego_prose_coherence.py --all --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fmt"))

from audit_lego_html import CHAPTER_LIST, CLASS, LEGO_MARK  # noqa: E402
from audit_prose import (  # noqa: E402
    CELL_END,
    CELL_START,
    INLINE_PY,
    _exec_python_cells,
    _resolve_ref,
)
from cell_exec import exec_cell_code, make_exec_namespace  # noqa: E402

GOAL = re.compile(r"#\s*[│├].*Goal:\s*(.+)", re.I)
GEMINI_MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 25


@dataclass
class ProsePacket:
    class_name: str
    cell_line: int
    goal: str
    snippets: list[dict]


def _chapter_qmd(root: Path, vol: str, name: str) -> Path:
    if name.startswith("appendix_"):
        return root / f"book/quarto/contents/{vol}/backmatter/{name}.qmd"
    return root / f"book/quarto/contents/{vol}/{name}/{name}.qmd"


def _lego_cell_lines(qmd: Path) -> dict[str, dict]:
    lines = qmd.read_text(encoding="utf-8").splitlines()
    out: dict[str, dict] = {}
    in_cell = False
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            buf = []
            start = i
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            code = "\n".join(buf)
            if not LEGO_MARK.search(code):
                continue
            goal_m = GOAL.search(code)
            goal = goal_m.group(1).strip() if goal_m else ""
            for cls in CLASS.findall(code):
                out[cls] = {"cell_line": start, "goal": goal}
            continue
        if in_cell:
            buf.append(line)
    return out


_INLINE_MATH = re.compile(r"(?<![\\])\$(?!\$)(.+?)(?<![\\])\$(?!\$)", re.DOTALL)
_DISPLAY_MATH = re.compile(r"(?<![\\])\$\$(.+?)(?<![\\])\$\$", re.DOTALL)
# Pandoc-style inline footnotes only — exclude LaTeX exponents/braces.
_FOOTNOTE = re.compile(r"\^[^\^\\{}[\]]+\^")


def _unwrap_outer_math_dollars(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$") and not s.startswith("$$"):
        return s[1:-1]
    return text


def _inside_inline_math(line: str, idx: int) -> bool:
    """True when ``idx`` sits inside an unclosed ``$...$`` span."""
    n = 0
    i = 0
    prefix = line[:idx]
    while i < len(prefix):
        if prefix[i] == "\\":
            i += 2
            continue
        if prefix[i : i + 2] == "$$":
            i += 2
            continue
        if prefix[i] == "$":
            n += 1
        i += 1
    return n % 2 == 1


def _substitute_python_ref(line: str, ref: str, ns: dict) -> str:
    """Replace ``{python} ref`` with resolved value, avoiding doubled ``$`` delimiters."""
    token = f"`{{python}} {ref}`"
    val = _resolve_ref(ref, ns)
    while True:
        idx = line.find(token)
        if idx == -1:
            break
        before = line[idx - 1] if idx > 0 else ""
        after = line[idx + len(token)] if idx + len(token) < len(line) else ""
        insert = val
        if _inside_inline_math(line, idx) or before == "$" or after == "$":
            insert = _unwrap_outer_math_dollars(insert)
        if before == "$" and insert.startswith("$") and not insert.startswith("$$"):
            insert = insert[1:]
        if after == "$" and insert.endswith("$") and not insert.endswith("$$"):
            insert = insert[:-1]
        line = line[:idx] + insert + line[idx + len(token) :]
    return line


def _strip_for_llm(text: str) -> str:
    """Light markdown strip for LLM review — preserve LaTeX math and currency ``\\$``."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)
    text = _FOOTNOTE.sub("", text)
    text = re.sub(r"\[@[^\]]+\]", "[cite]", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = _DISPLAY_MATH.sub(r"\1", text)
    prev = None
    while prev != text:
        prev = text
        text = _INLINE_MATH.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_packets(qmd: Path) -> list[ProsePacket]:
    lines = qmd.read_text(encoding="utf-8").splitlines()
    meta = _lego_cell_lines(qmd)
    ns = _exec_python_cells(lines)
    grouped: dict[str, list[dict]] = {k: [] for k in meta}

    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if in_cell:
            continue

        refs = INLINE_PY.findall(line)
        if not refs:
            continue
        rendered = line
        for ref in refs:
            rendered = _substitute_python_ref(rendered, ref, ns)
        snippet = _strip_for_llm(rendered)
        if len(snippet) < 20:
            continue
        for ref in refs:
            cls = ref.split(".")[0]
            if cls not in grouped:
                continue
            grouped[cls].append({"line": i, "text": snippet, "refs": refs})

    packets: list[ProsePacket] = []
    for cls, snippets in grouped.items():
        if not snippets:
            continue
        info = meta.get(cls, {"cell_line": 0, "goal": ""})
        # Dedupe near-identical snippets on same line
        seen: set[int] = set()
        uniq = []
        for s in snippets:
            if s["line"] in seen:
                continue
            seen.add(s["line"])
            uniq.append(s)
        packets.append(
            ProsePacket(
                class_name=cls,
                cell_line=info["cell_line"],
                goal=info["goal"],
                snippets=uniq[:8],
            )
        )
    return packets


def _call_gemini(prompt: str, *, timeout: int = 600) -> dict | None:
    r = subprocess.run(
        ["gemini", "-m", GEMINI_MODEL, "-p", prompt, "--yolo", "--skip-trust"],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[4]),
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip() or f"exit {r.returncode}"}
    s = r.stdout
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        return {"error": "no JSON in response", "raw": s[:500]}
    try:
        return json.loads(s[i : j + 1])
    except json.JSONDecodeError as exc:
        return {"error": str(exc), "raw": s[i : j + 1][:500]}


def _judge_batch(chapter_slug: str, batch: list[ProsePacket]) -> dict:
    items = []
    for p in batch:
        items.append(
            {
                "id": p.class_name,
                "cell_line": p.cell_line,
                "goal": p.goal,
                "snippets": p.snippets,
            }
        )
    prompt = f"""You are reviewing inline-computed numbers in a machine learning systems textbook chapter.

Chapter: {chapter_slug}

For each LEGO block below, the prose has Python-computed values already substituted into the text. Judge whether the numbers and units make sense together and the sentence reads coherently for a technical reader. Flag only real problems: wrong units, contradictory numbers, broken grammar after substitution, or values that cannot plausibly fit the claim.

Respond with a single JSON object (no markdown fences):
{{
  "chapter": "{chapter_slug}",
  "items": [
    {{
      "id": "<class name>",
      "verdict": "pass" | "warn" | "fail",
      "issues": [
        {{"line": <int>, "severity": "warn"|"fail", "note": "<one sentence>"}}
      ]
    }}
  ]
}}

Include every id from the input. Use verdict "pass" when coherent. Empty issues array when pass.

INPUT:
{json.dumps(items, indent=2)}
"""
    result = _call_gemini(prompt)
    if result is None:
        return {"chapter": chapter_slug, "error": "gemini call failed"}
    return result


def review_chapter(root: Path, vol: str, name: str) -> dict:
    slug = f"{vol}/{name}"
    qmd = _chapter_qmd(root, vol, name)
    if not qmd.is_file():
        return {"chapter": slug, "status": "NO_QMD"}

    try:
        meta = _lego_cell_lines(qmd)
        packets = extract_packets(qmd)
    except RuntimeError as exc:
        return {"chapter": slug, "status": "EXEC_FAIL", "error": str(exc)}

    lego_classes = sorted(meta.keys())
    if not packets:
        return {
            "chapter": slug,
            "status": "NO_PACKETS",
            "packets": 0,
            "lego_classes": lego_classes,
            "missing_coverage": lego_classes,
        }

    all_items: list[dict] = []
    errors: list[str] = []
    for i in range(0, len(packets), BATCH_SIZE):
        batch = packets[i : i + BATCH_SIZE]
        result = _judge_batch(slug, batch)
        if "error" in result and "items" not in result:
            errors.append(str(result["error"]))
            continue
        all_items.extend(result.get("items", []))

    fails = sum(1 for it in all_items if it.get("verdict") == "fail")
    warns = sum(1 for it in all_items if it.get("verdict") == "warn")
    status = "FAIL" if fails or errors else ("WARN" if warns else "PASS")
    reported_ids = {it.get("id") for it in all_items}
    missing_coverage = sorted(set(lego_classes) - reported_ids)
    return {
        "chapter": slug,
        "status": status,
        "packets": len(packets),
        "lego_classes": lego_classes,
        "missing_coverage": missing_coverage,
        "items": all_items,
        "errors": errors,
    }


def _review_chapter_worker(args: tuple[str, str, str]) -> dict:
    root_s, vol, name = args
    return review_chapter(Path(root_s), vol, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Review all 44 chapters")
    parser.add_argument("--chapter", help="Single chapter slug vol1/name")
    parser.add_argument("--workers", type=int, default=2, help="Parallel chapter reviews (processes)")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("book/tools/audit/artifacts/lego_prose_coherence_report.json"),
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail if any LEGO class from cells is missing from the LLM report",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit 1 for WARN verdicts as well as FAIL",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    if args.chapter:
        vol, name = args.chapter.split("/", 1)
        chapters = [(vol, name)]
    elif args.all:
        chapters = [(vol, name) for vol, names in CHAPTER_LIST.items() for name in names]
    else:
        parser.error("Specify --chapter or --all")
        return 2

    report: list[dict] = []
    if args.workers <= 1 or len(chapters) == 1:
        for vol, name in chapters:
            report.append(review_chapter(root, vol, name))
    else:
        root_s = str(root)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_review_chapter_worker, (root_s, vol, name)): (vol, name)
                for vol, name in chapters
            }
            for fut in as_completed(futs):
                report.append(fut.result())
        report.sort(key=lambda r: r.get("chapter", ""))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ch_pass = sum(1 for r in report if r.get("status") == "PASS")
    ch_fail = sum(1 for r in report if r.get("status") == "FAIL")
    ch_warn = sum(1 for r in report if r.get("status") == "WARN")
    coverage_failures: list[tuple[str, list[str]]] = []
    if args.strict_coverage:
        for r in report:
            missing = r.get("missing_coverage") or []
            if missing:
                coverage_failures.append((r["chapter"], missing))

    print("LEGO prose coherence (LLM review)")
    print("=" * 72)
    print(f"Chapters: {len(report)} | PASS: {ch_pass} | WARN: {ch_warn} | FAIL: {ch_fail}")
    print(f"Report: {args.report}\n")
    for r in report:
        if r.get("status") == "PASS" and not (
            args.strict_coverage and r.get("missing_coverage")
        ):
            print(f"PASS {r['chapter']} ({r.get('packets', 0)} LEGO blocks)")
            continue
        if r.get("missing_coverage"):
            print(
                f"COVERAGE {r['chapter']}: missing LLM items for "
                f"{', '.join(r['missing_coverage'])}"
            )
        if r.get("status") != "PASS":
            print(f"{r.get('status', '?')} {r['chapter']}: {r.get('error', r.get('errors', ''))}")
        for it in r.get("items", []):
            if it.get("verdict") in ("fail", "warn"):
                for issue in it.get("issues", []):
                    print(
                        f"  {it['id']} L{issue.get('line')}: "
                        f"[{issue.get('severity')}] {issue.get('note')}"
                    )

    if coverage_failures:
        print("\nStrict coverage failures:")
        for chapter, missing in coverage_failures:
            print(f"  {chapter}: {', '.join(missing)}")

    if ch_fail or coverage_failures:
        return 1
    if args.fail_on_warn and ch_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
