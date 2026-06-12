#!/usr/bin/env python3
"""
audit_prose.py
==============

Exec a chapter's Python cells, substitute ``{python} Class.var_str`` inline
refs with rendered values, and print prose previews — without a full HTML build.

Use this while tuning fmt precision: read the sentence aloud, then confirm
with ``./book/binder build <chapter>`` + ``audit_html.py`` as the final check.

See ``fmt/README.md`` for the full chapter workflow.

Usage::

    PYTHONPATH=mlsysim python3 book/tools/audit/fmt/audit_prose.py \\
        book/quarto/contents/vol1/introduction/introduction.qmd

    # Only lines that use a given LEGO class:
    ... introduction.qmd --class ResNet50DamExample

    # Machine-readable:
    ... introduction.qmd --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_exec import exec_cell_code, make_exec_namespace, setup_headless_matplotlib
from spurious_zero import SPURIOUS_ZERO, is_spurious_zero_false_positive

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
setup_headless_matplotlib()

INLINE_PY = re.compile(
    r"`\{python\}\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*(?:\[-?\d+\])?)*)`"
)
REF_PART = re.compile(r"^([A-Za-z_]\w*)(?:\[(-?\d+)\])?$")


@dataclass
class ProsePreview:
    line: int
    refs: list[str]
    preview: str
    flags: list[str]


def _exec_python_cells(lines: list[str]) -> dict:
    """Exec all Quarto python cells in document order (shared namespace)."""
    ns = make_exec_namespace()
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
            try:
                exec_cell_code(code, ns)
            except Exception as exc:
                ns["_last_exec_error"] = exc
                raise RuntimeError(
                    f"Python cell failed during prose preview: {exc}"
                ) from exc
            continue
        if in_cell:
            buf.append(line)
    return ns


def _resolve_ref(ref: str, ns: dict) -> str:
    """Resolve ``Class.attr`` or bare ``name`` from the exec namespace."""
    parts = ref.split(".")
    try:
        head = REF_PART.match(parts[0])
        if not head:
            return f"<MISSING:{ref}>"
        val = ns[head.group(1)]
        if head.group(2) is not None:
            val = val[int(head.group(2))]
        for part in parts[1:]:
            match = REF_PART.match(part)
            if not match:
                return f"<MISSING:{ref}>"
            val = getattr(val, match.group(1))
            if match.group(2) is not None:
                val = val[int(match.group(2))]
    except (KeyError, AttributeError, IndexError, TypeError):
        return f"<MISSING:{ref}>"

    # MarkdownStr / fmt outputs: use plain string form
    return str(val)


def _strip_light_markdown(text: str) -> str:
    """Remove lightweight markdown so previews read like prose."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\^[^^]+\^", "", text)
    text = re.sub(r"\[@[^\]]+\]", "[cite]", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\$\$.*?\$\$", "[math]", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "[math]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _flag_preview(preview: str) -> list[str]:
    flags: list[str] = []
    for m in SPURIOUS_ZERO.finditer(preview):
        start = max(0, m.start() - 20)
        end = min(len(preview), m.end() + 20)
        ctx = preview[start:end]
        if is_spurious_zero_false_positive(ctx):
            continue
        flags.append("spurious_.0")
        break
    if "<MISSING:" in preview:
        flags.append("missing_ref")
    return flags


def audit_prose_previews(
    qmd_path: Path,
    *,
    class_filter: str | None = None,
) -> list[ProsePreview]:
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    ns = _exec_python_cells(lines)
    out: list[ProsePreview] = []

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
        if class_filter and not any(
            ref.split(".")[0] == class_filter for ref in refs
        ):
            continue

        preview = line
        for ref in refs:
            val = _resolve_ref(ref, ns)
            preview = preview.replace(f"`{{python}} {ref}`", val)

        preview = _strip_light_markdown(preview)
        flags = _flag_preview(preview)
        out.append(ProsePreview(line=i, refs=refs, preview=preview, flags=flags))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qmd", type=Path, help="Chapter .qmd file")
    parser.add_argument(
        "--class",
        dest="class_filter",
        metavar="NAME",
        help="Only show lines referencing this LEGO class (e.g. ResNet50DamExample)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--flagged-only",
        action="store_true",
        help="Only show lines with spurious .0 or missing refs",
    )
    args = parser.parse_args()

    if not args.qmd.is_file():
        print(f"File not found: {args.qmd}", file=sys.stderr)
        return 1

    try:
        previews = audit_prose_previews(
            args.qmd, class_filter=args.class_filter
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.flagged_only:
        previews = [p for p in previews if p.flags]

    if args.json:
        print(json.dumps([asdict(p) for p in previews], indent=2))
        return 0

    print(f"Prose preview: {args.qmd} ({len(previews)} inline-python lines)\n")
    for p in previews:
        flag = f"  ⚠ {','.join(p.flags)}" if p.flags else ""
        print(f"L{p.line:4d}{flag}")
        print(f"  {p.preview}\n")

    flagged = sum(1 for p in previews if p.flags)
    if flagged:
        print(f"Flagged: {flagged} line(s)")
        return 1
    print("No spurious .0 or missing refs in previews.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
