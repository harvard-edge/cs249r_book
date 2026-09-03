#!/usr/bin/env python3
"""Split python cells that define more than one LEGO class into separate cells."""
from __future__ import annotations

import re
import sys
from pathlib import Path

CLASS = re.compile(r"^class\s+(\w+)", re.M)
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
INLINE = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")


def first_ref_line(content: str, cls: str) -> int | None:
    pat = re.compile(rf"`\{{python\}}\s+{re.escape(cls)}\.")
    m = pat.search(content)
    if not m:
        return None
    return content[: m.start()].count("\n") + 1


def find_cell_at_line(lines: list[str], line_no: int) -> tuple[int, int]:
    """Return (start_idx, end_idx) for cell containing 1-based line_no."""
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            start = i
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                j += 1
            if start + 1 <= line_no <= j + 1:
                return start, j
            i = j + 1
        else:
            i += 1
    raise ValueError(f"No cell found at line {line_no}")


def split_cell_block(block_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Split cell into header+first class, then one block per additional class."""
    text = "\n".join(block_lines)
    matches = list(CLASS.finditer(text))
    if len(matches) <= 1:
        return block_lines, []

    # Split on class boundaries inside the cell body (skip fence lines)
    body_start = 1  # after ```{python}
    body_end = len(block_lines) - 1  # before closing ```
    body = block_lines[body_start:body_end]
    body_text = "\n".join(body)

    class_starts: list[tuple[str, int]] = []
    for m in CLASS.finditer(body_text):
        class_starts.append((m.group(1), m.start()))

    first_cls_start = class_starts[0][1]
    header = body_text[:first_cls_start].rstrip("\n")
    if header:
        header += "\n"

    chunks: list[str] = []
    for idx, (cls, pos) in enumerate(class_starts):
        end = class_starts[idx + 1][1] if idx + 1 < len(class_starts) else len(body_text)
        chunks.append(body_text[pos:end].rstrip("\n"))

    first_cell_body = header + chunks[0]
    first_cell = [block_lines[0], *first_cell_body.splitlines(), block_lines[-1]]

    extra_cells: list[list[str]] = []
    for chunk in chunks[1:]:
        extra_body = header + chunk
        extra_cells.append([block_lines[0], *extra_body.splitlines(), block_lines[-1]])

    return first_cell, extra_cells


def insert_before_line(lines: list[str], insert_at: int, cell_lines: list[str]) -> None:
    """Insert cell block before 1-based insert_at."""
    idx = insert_at - 1
    insertion = cell_lines if idx == 0 or lines[idx - 1].strip() else [""] + cell_lines
    lines[idx:idx] = insertion


def process_file(path: Path, cell_lines_to_split: list[int], relocate: dict[str, int] | None = None) -> bool:
    relocate = relocate or {}
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    changed = False

    # Process from bottom to top so line numbers stay valid
    for line_no in sorted(cell_lines_to_split, reverse=True):
        start, end = find_cell_at_line(lines, line_no)
        block = lines[start : end + 1]
        classes = CLASS.findall("\n".join(block))
        if len(classes) <= 1:
            continue

        first_cell, extra_cells = split_cell_block(block)
        lines[start : end + 1] = first_cell
        changed = True

        # Insert extra cells: relocate if requested, else immediately after first
        insert_pos = end + 2  # 1-based line after first cell
        for cls_name, extra in zip(classes[1:], extra_cells):
            target = relocate.get(cls_name)
            if target:
                insert_before_line(lines, target, extra)
            else:
                idx = insert_pos - 1
                lines[idx:idx] = extra + [""]
                insert_pos += len(extra) + 1

    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def remove_class_from_cell(path: Path, cell_line: int, remove_class: str) -> bool:
    """Remove a duplicate class definition from a cell (keep other classes)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = find_cell_at_line(lines, cell_line)
    block = lines[start : end + 1]
    body = "\n".join(block[1:-1])

    pat = re.compile(
        rf"(# ┌── LEGO ──.*?\n)?class {re.escape(remove_class)}:.*?(?=\n# ┌── LEGO|\nclass |\Z)",
        re.S,
    )
    new_body, n = pat.subn("", body)
    if n == 0:
        return False
    new_body = re.sub(r"\n{3,}", "\n\n", new_body.strip("\n"))
    lines[start : end + 1] = [block[0], *new_body.splitlines(), block[-1]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


JOBS = [
    # (path, cell_start_lines, {class: insert_before_line})
    (
        "book/quarto/contents/vol1/ml_ops/ml_ops.qmd",
        [187],
        {"RetrainingAnchor": 1318},
    ),
    (
        "book/quarto/contents/vol1/training/training.qmd",
        [269, 3548],
        {"TrainingHardware": 563},
    ),
    (
        "book/quarto/contents/vol1/ml_systems/ml_systems.qmd",
        [2046],
        {"DataLocalityInvariant": 2403},
    ),
    (
        "book/quarto/contents/vol2/security_privacy/security_privacy.qmd",
        [1261],
        {"TEEMemoryFootprint": 2566},
    ),
    (
        "book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd",
        [1972],
        {},
    ),
    ("book/quarto/contents/vol1/model_serving/model_serving.qmd", [2156], {}),
    ("book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd", [2958], {}),
    ("book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd", [324], {}),
    ("book/quarto/contents/vol2/data_storage/data_storage.qmd", [476, 1844, 1922, 1991, 2470], {}),
    ("book/quarto/contents/vol2/distributed_training/distributed_training.qmd", [1459, 2078], {}),
    ("book/quarto/contents/vol2/inference/inference.qmd", [1118], {}),
    (
        "book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd",
        [1806, 2281],
        {},
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    frameworks = root / "book/quarto/contents/vol1/frameworks/frameworks.qmd"
    if remove_class_from_cell(frameworks, 48, "GraphOptimizationStats"):
        print(f"removed duplicate GraphOptimizationStats from {frameworks}")

    for rel, cell_lines, relocate in JOBS:
        path = root / rel
        if process_file(path, cell_lines, relocate):
            print(f"split cells in {path}")

    # Verify no multi-class cells remain
    remaining = []
    for qmd in sorted((root / "book/quarto/contents").rglob("*.qmd")):
        content = qmd.read_text(encoding="utf-8")
        if "`{python}" not in content:
            continue
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            if CELL_START.match(lines[i]):
                start = i + 1
                j = i + 1
                while j < len(lines) and not CELL_END.match(lines[j]):
                    j += 1
                block = "\n".join(lines[i : j + 1])
                classes = CLASS.findall(block)
                if len(classes) > 1:
                    remaining.append((str(qmd), start, classes))
                i = j + 1
            else:
                i += 1

    if remaining:
        print("REMAINING multi-class cells:")
        for item in remaining:
            print(f"  {item}")
        sys.exit(1)
    print("All multi-class cells split successfully.")


if __name__ == "__main__":
    main()
