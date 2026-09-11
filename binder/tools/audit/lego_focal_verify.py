#!/usr/bin/env python3
"""Verify LEGO focal-point coherence for one or more QMD chapters."""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

INLINE = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
CLASS = re.compile(r"^class\s+(\w+)", re.M)
EXPORT = re.compile(r"^\s+(\w+(?:_str|_math|_eq|_frac))\s*=", re.M)
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
H2 = re.compile(r"^##\s+")
SCOPE_CHAPTER_ANCHOR = re.compile(r"#\s*│\s*Scope:\s*chapter-anchor\b", re.I)
ANCHOR_HEADER_HINT = re.compile(
    r"(?i)"
    r"(?:chapter[- ]wide|and chapter summary|chapter anchor|"
    r"running case(?:[- ]study)?|lighthouse profile recap|"
    r"chapter takeaway|chapter-wide)"
)


def _is_chapter_anchor(block: str) -> bool:
    """True when the cell header documents an intentional chapter-wide anchor.

    Chapter anchors keep one scenario's numbers consistent across a problem
    callout, a later walkthrough, and a summary bullet. That legitimately
    spans sections; it is not the unrelated mega-class anti-pattern.
    """
    if SCOPE_CHAPTER_ANCHOR.search(block):
        return True
    header = block.split("class ", 1)[0]
    return bool(ANCHOR_HEADER_HINT.search(header))


def analyze(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    if "`{python}" not in content:
        return {"path": str(path), "skipped": True}

    lines = content.splitlines()
    cells: list[tuple[int, int, str, str | None]] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            start = i + 1
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                j += 1
            block = "\n".join(lines[i : j + 1])
            m = CLASS.search(block)
            cells.append((i + 1, j + 1, block, m.group(1) if m else None))
            i = j + 1
        else:
            i += 1

    class_cell = {cls: start for start, _, _, cls in cells if cls}
    class_cell_end = {cls: end for _, end, _, cls in cells if cls}
    class_block = {cls: block for _, _, block, cls in cells if cls}

    cross: dict[str, list[str]] = defaultdict(list)
    for _, _, block, cls in cells:
        if not cls:
            continue
        for other in class_cell:
            if other != cls and re.search(rf"\b{re.escape(other)}\.\w+", block):
                cross[other].append(cls)

    cross_cell_total = sum(len(v) for v in cross.values())
    cross_cell_violations = sum(
        len(v)
        for target, v in cross.items()
        if not _is_chapter_anchor(class_block[target])
    )

    refs: dict[str, list[int]] = defaultdict(list)
    for m in INLINE.finditer(content):
        cls = m.group(1).split(".")[0]
        refs[cls].append(content[: m.start()].count("\n") + 1)

    def section_at(line_no: int) -> str:
        for idx in range(line_no - 1, -1, -1):
            if H2.match(lines[idx]):
                return lines[idx].strip()
        return "(preamble)"

    issues: list[dict] = []
    ok = 0
    for cls, cell_line in sorted(class_cell.items(), key=lambda x: x[1]):
        block = class_block[cls]
        exports = len(EXPORT.findall(block))
        ref_lines = sorted(set(refs.get(cls, [])))
        if not ref_lines:
            issues.append({"class": cls, "kind": "no_prose_refs"})
            continue
        span = ref_lines[-1] - ref_lines[0]
        sections = {section_at(ln) for ln in ref_lines}
        gap = ref_lines[0] - class_cell_end[cls]
        flags = []
        anchor = _is_chapter_anchor(block)
        if cross.get(cls) and not anchor:
            flags.append(f"cross_cell:{','.join(cross[cls])}")
        if not anchor and len(sections) > 1 and span > 80:
            flags.append(f"multi_section:{len(sections)}")
        if not anchor and span > 200:
            flags.append(f"span:{span}")
        if not anchor and gap > 150:
            flags.append(f"gap:{gap}")
        if flags:
            issues.append({"class": cls, "kind": "coherence", "flags": flags, "exports": exports})
        else:
            ok += 1

    return {
        "path": str(path),
        "classes": len(class_cell),
        "ok": ok,
        "issues": issues,
        "cross_cell_total": cross_cell_total,
        "cross_cell_violations": cross_cell_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="QMD files or directories")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.qmd")))
        else:
            files.append(p)

    failed = False
    for f in files:
        r = analyze(f)
        if r.get("skipped"):
            continue
        rel = f
        n_issues = len(r["issues"])
        status = "OK" if n_issues == 0 and r["cross_cell_violations"] == 0 else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"{status} {rel}  ok={r['ok']}/{r['classes']}  cross={r['cross_cell_total']}  issues={n_issues}")
        for issue in r["issues"]:
            print(f"  - {issue['class']}: {issue.get('flags') or issue.get('kind')}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
