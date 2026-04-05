#!/usr/bin/env python3
"""
Sort applicability matrix for lookup: within each of the three panels (columns),
category blocks (bold headers) are ordered A--Z, and topic rows are A--Z within
each block. Panels are filled independently and padded to equal height, so a
given table row is not meant to align semantically across columns.

Prefer editing applicability_matrix.yaml and running generate_app_matrix_from_yaml.py
for a stable source of truth; use this script only to re-sort an existing app_matrix.tex
without going through YAML.

Run from interviews/paper: python3 sort_app_matrix.py
"""

from __future__ import annotations

import re
from pathlib import Path

TEX = Path(__file__).resolve().parent / "app_matrix.tex"
MASK = "green!60!black"
MASK_TOKEN = "greenBLACKNOSPLIT"
IMPLICIT = "__IMPLICIT_HEADER__"


def split_row(line: str) -> list[str]:
    """15 cells. Mask \\& and color names so & can split safely."""
    line2 = line.replace(r"\&", "__AMP__")
    line2 = line2.replace(MASK, MASK_TOKEN)
    parts = re.split(r"\s*&\s*", line2)
    if len(parts) != 15:
        raise ValueError(f"Expected 15 cells, got {len(parts)}: {line[:100]!r}...")
    return [
        p.replace(MASK_TOKEN, MASK).replace("__AMP__", r"\&") for p in parts
    ]


def join_row(parts: list[str]) -> str:
    return " & ".join(parts) + r" \\"


def category_sort_key(header_line: str) -> str:
    """Sort key from \\textbf{Name (n)} -> name (lowercase)."""
    m = re.search(r"\\textbf\{([^}]+)\}", header_line)
    if not m:
        return header_line.lower()
    inner = m.group(1)
    m2 = re.match(r"(.+?)\s*\(\d+\)\s*$", inner)
    name = (m2.group(1) if m2 else inner).strip()
    return name.lower()


def topic_sort_key(topic: str) -> str:
    return topic.strip().replace(r"\&", "&").lower()


def parse_columns(lines: list[str]) -> list[list[dict]]:
    columns: list[list[dict]] = [[], [], []]
    cur: list[int | None] = [None, None, None]

    def ensure_cat(col: int, header: str) -> None:
        columns[col].append({"header": header, "topics": []})
        cur[col] = len(columns[col]) - 1

    for line in lines:
        if not line.endswith(r"\\"):
            continue
        line_stripped = line[:-2].strip()
        parts = split_row(line_stripped)
        for col in range(3):
            cell = parts[col * 5]
            marks = parts[col * 5 + 1 : col * 5 + 5]
            if cell.startswith("\\textbf{"):
                ensure_cat(col, cell)
            elif cell.strip() == "" and all(m.strip() == "" for m in marks):
                continue
            else:
                if cur[col] is None:
                    ensure_cat(col, IMPLICIT)
                columns[col][cur[col]]["topics"].append((cell, marks))
    return columns


def sort_columns(columns: list[list[dict]]) -> None:
    for col in range(3):
        blocks = columns[col]
        for b in blocks:
            if b["header"] == IMPLICIT:
                b["_sort_cat"] = "data"
            else:
                b["_sort_cat"] = category_sort_key(b["header"])
        blocks.sort(key=lambda b: b["_sort_cat"])
        for b in blocks:
            b.pop("_sort_cat", None)
            b["topics"].sort(key=lambda tm: topic_sort_key(tm[0]))


def flatten_col(blocks: list[dict]) -> list[tuple]:
    rows: list[tuple] = []
    for b in blocks:
        if b["header"] != IMPLICIT:
            rows.append(("header", b["header"]))
        for topic, marks in b["topics"]:
            rows.append(("topic", topic, marks))
    return rows


def emit_table_body(flat: list[list[tuple]]) -> str:
    max_len = max(len(x) for x in flat)
    out_lines: list[str] = []
    for i in range(max_len):
        parts: list[str] = []
        for col in range(3):
            if i >= len(flat[col]):
                parts.extend(["", "", "", "", ""])
                continue
            item = flat[col][i]
            if item[0] == "header":
                parts.extend([item[1], "", "", "", ""])
            elif item[0] == "topic":
                parts.extend([item[1]] + item[2])
            else:
                parts.extend(["", "", "", "", ""])
        out_lines.append(join_row(parts))
    return "\n".join(out_lines)


def main() -> None:
    text = TEX.read_text()
    pre, rest = text.split(r"\midrule", 1)
    body, post = rest.split(r"\bottomrule", 1)
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]

    columns = parse_columns(lines)
    sort_columns(columns)
    flat = [flatten_col(columns[c]) for c in range(3)]
    max_len = max(len(x) for x in flat)
    for c in range(3):
        while len(flat[c]) < max_len:
            flat[c].append(("empty",))

    new_body = emit_table_body(flat)
    new_tex = pre + r"\midrule" + "\n" + new_body + "\n" + r"\bottomrule" + post
    TEX.write_text(new_tex)
    print(
        f"Wrote {TEX} ({max_len} rows; categories A–Z and topics A–Z within each panel)"
    )


if __name__ == "__main__":
    main()
