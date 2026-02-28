#!/usr/bin/env python3
"""Migrate LEGO bridge variables to direct ClassName.attr access.

For each .qmd file, finds bridge assignments like:
    var = ClassName.attr
and replaces all downstream uses (inline refs + Python code) with
the direct ClassName.attr access, then removes the bridge line.
"""

import re
import sys
from pathlib import Path

BOOK_ROOT = Path("/Users/VJ/GitHub/mlsysbook-vols/book/quarto")
CONTENTS = BOOK_ROOT / "contents"

CODE_BLOCK_START = re.compile(r"^```\{python\}")
CODE_BLOCK_END = re.compile(r"^```\s*$")
BRIDGE_RE = re.compile(r"^(\w+)\s*=\s*(\w+\.\w+)\s*(?:#.*)?$")
CLASS_DEF_RE = re.compile(r"^class\s+(\w+)\s*[:(]")
REASSIGN_RE = re.compile(r"^(\w+)\s*=")
AUGMENTED_RE = re.compile(
    r"^(\w+)\s*(?:\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|<<=|>>=)"
)
EXPORT_COMMENT_PATTERNS = [
    re.compile(r"^#\s*[┌├└─╌]+"),
    re.compile(r"^#\s*EXPORTS", re.I),
    re.compile(r"^#\s*Expose\s+only\b", re.I),
    re.compile(r"^#\s*Math\s+vars?\s+needed\b", re.I),
    re.compile(r"^#\s*Public\s+API", re.I),
    re.compile(r"^#\s*Bridge\b", re.I),
]


def _is_export_comment(line: str) -> bool:
    stripped = line.strip()
    return any(p.match(stripped) for p in EXPORT_COMMENT_PATTERNS)


def _find_code_blocks(lines: list[str]) -> list[tuple[int, int]]:
    blocks = []
    in_block = False
    start = -1
    for i, line in enumerate(lines):
        if CODE_BLOCK_START.match(line) and not in_block:
            in_block = True
            start = i
        elif in_block and CODE_BLOCK_END.match(line):
            blocks.append((start, i))
            in_block = False
    return blocks


def _in_code_block(idx: int, blocks: list[tuple[int, int]]) -> bool:
    for s, e in blocks:
        if s < idx < e:
            return True
    return False


def _find_class_body_lines(lines, blocks):
    body = set()
    for bs, be in blocks:
        cls_start = None
        for i in range(bs + 1, be):
            stripped = lines[i].strip()
            is_indented = lines[i][:1].isspace() and stripped
            if CLASS_DEF_RE.match(stripped) and not lines[i][:1].isspace():
                cls_start = i
            elif cls_start is not None:
                if is_indented:
                    body.add(i)
                elif stripped and not lines[i][:1].isspace():
                    cls_start = None
                    if CLASS_DEF_RE.match(stripped):
                        cls_start = i
    return body


def _find_bridges(lines, blocks, class_body):
    bridges = []
    for bs, be in blocks:
        for i in range(bs + 1, be):
            if i in class_body:
                continue
            line = lines[i]
            stripped = line.strip()
            if line[:1].isspace() and stripped:
                continue
            m = BRIDGE_RE.match(stripped)
            if not m:
                continue
            var, attr = m.group(1), m.group(2)
            cls = attr.split(".")[0]
            if not cls[:1].isupper():
                continue
            bridges.append({"var": var, "attr": attr, "line": i})
    return bridges


def _scope_bridges(bridges, lines, blocks):
    all_assigns: dict[str, list[int]] = {}
    for bs, be in blocks:
        for i in range(bs + 1, be):
            stripped = lines[i].strip()
            if lines[i][:1].isspace() and stripped:
                continue
            m = REASSIGN_RE.match(stripped)
            if m:
                all_assigns.setdefault(m.group(1), []).append(i)

    for b in bridges:
        scope_end = len(lines) - 1
        for a in sorted(all_assigns.get(b["var"], [])):
            if a > b["line"]:
                scope_end = a - 1
                break
        b["scope_end"] = scope_end


def _safety_check(bridges, lines, blocks, class_body):
    for b in bridges:
        b["safe"] = True
        b["skip_reason"] = None
        for bs, be in blocks:
            for i in range(bs + 1, be):
                if i <= b["line"] or i > b["scope_end"]:
                    continue
                if i in class_body:
                    continue
                stripped = lines[i].strip()
                if lines[i][:1].isspace() and stripped:
                    continue
                m = AUGMENTED_RE.match(stripped)
                if m and m.group(1) == b["var"]:
                    b["safe"] = False
                    b["skip_reason"] = f"augmented assignment at line {i + 1}"
                    break
            if not b["safe"]:
                break


def process_file(filepath, dry_run=False):
    text = filepath.read_text()
    lines = text.split("\n")

    blocks = _find_code_blocks(lines)
    class_body = _find_class_body_lines(lines, blocks)
    bridges = _find_bridges(lines, blocks, class_body)

    if not bridges:
        return False, {"bridges": 0}

    _scope_bridges(bridges, lines, blocks)
    _safety_check(bridges, lines, blocks, class_body)

    safe = [b for b in bridges if b["safe"]]
    unsafe = [b for b in bridges if not b["safe"]]

    new = list(lines)
    remove = set()
    stats = {
        "bridges": len(bridges),
        "removed": 0,
        "inline": 0,
        "code": 0,
        "lines_removed": 0,
        "skipped": [
            f"  line {b['line']+1}: {b['var']} = {b['attr']} ({b['skip_reason']})"
            for b in unsafe
        ],
    }

    for b in safe:
        var, attr, start, end = b["var"], b["attr"], b["line"], b["scope_end"]
        remove.add(start)
        stats["removed"] += 1

        var_re = re.compile(r"\b" + re.escape(var) + r"\b")
        inline_re = re.compile(r"`\{python\}\s+" + re.escape(var) + r"`")

        for i in range(start + 1, min(end + 1, len(new))):
            if i in remove or i in class_body:
                continue
            old = new[i]
            if _in_code_block(i, blocks):
                new[i] = var_re.sub(attr, new[i])
                if new[i] != old:
                    stats["code"] += 1
            else:
                new[i] = inline_re.sub(f"`{{python}} {attr}`", new[i])
                if new[i] != old:
                    stats["inline"] += 1

    for idx in sorted(remove):
        j = idx - 1
        while j >= 0:
            stripped = new[j].strip()
            if not stripped:
                remove.add(j)
                j -= 1
                continue
            if _is_export_comment(new[j]):
                remove.add(j)
                j -= 1
            else:
                break

    stats["lines_removed"] = len(remove)

    result = []
    prev_blank = False
    for i, line in enumerate(new):
        if i in remove:
            continue
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank

    new_text = "\n".join(result)
    modified = new_text != text
    if modified and not dry_run:
        filepath.write_text(new_text)
    return modified, stats


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    files = sorted(CONTENTS.rglob("*.qmd"))
    totals = dict(
        processed=0, modified=0, bridges=0, removed=0,
        inline=0, code=0, lines_removed=0, skipped=[],
    )

    for fp in files:
        mod, st = process_file(fp, dry_run)
        totals["processed"] += 1
        totals["bridges"] += st.get("bridges", 0)
        if mod:
            totals["modified"] += 1
            totals["removed"] += st.get("removed", 0)
            totals["inline"] += st.get("inline", 0)
            totals["code"] += st.get("code", 0)
            totals["lines_removed"] += st.get("lines_removed", 0)
            rel = fp.relative_to(BOOK_ROOT)
            tag = "[DRY] " if dry_run else ""
            print(
                f"  {tag}✓ {rel}: "
                f"{st['removed']} bridges, "
                f"{st['inline']} inline, "
                f"{st['code']} code, "
                f"{st['lines_removed']} lines"
            )
            if st.get("skipped"):
                for s in st["skipped"]:
                    totals["skipped"].append(f"{rel}: {s}")

    print(f"\n{'='*60}")
    print(f"{'DRY RUN ' if dry_run else ''}Migration Summary")
    print(f"{'='*60}")
    print(f"Files processed:      {totals['processed']}")
    print(f"Files modified:       {totals['modified']}")
    print(f"Bridges found:        {totals['bridges']}")
    print(f"Bridges removed:      {totals['removed']}")
    print(f"Inline refs updated:  {totals['inline']}")
    print(f"Code refs updated:    {totals['code']}")
    print(f"Lines removed:        {totals['lines_removed']}")

    if totals["skipped"]:
        print(f"\nSkipped (unsafe — augmented assignment):")
        for s in totals["skipped"]:
            print(f"  ⚠ {s}")


if __name__ == "__main__":
    main()
