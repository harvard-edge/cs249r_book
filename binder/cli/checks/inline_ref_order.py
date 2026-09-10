"""
Catch inline `{python}` references that run before the cell that defines them.

Quarto executes a chapter's cells top to bottom, so a prose reference to
`SomeClass.value_str` fails at render time unless the LEGO cell defining
`SomeClass` appears earlier in the file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

CELL = re.compile(r"^```\{python\}\s*$(.*?)^```\s*$", re.M | re.S)
CLASSDEF = re.compile(r"^\s*class\s+(\w+)", re.M)
ASSIGN = re.compile(r"^\s*(\w+)\s*=", re.M)
INLINE = re.compile(r"`\{python\}\s*([A-Za-z_]\w*)(?:\.\w+)*\s*`")


@dataclass
class InlineOrderIssue:
    path: Path
    line: int
    name: str
    raw: str
    message: str


def check_inline_order(path: Path, text: str) -> List[InlineOrderIssue]:
    """Check if any inline {python} references occur before their defining cell."""
    events: List[Tuple[int, str, object]] = []
    for m in CELL.finditer(text):
        events.append((m.start(), "cell", m.group(1)))
    for m in INLINE.finditer(text):
        events.append((m.start(), "ref", (m.group(1), m.group(0))))

    cell_spans = [(m.start(), m.end()) for m in CELL.finditer(text)]

    def inside_cell(pos: int) -> bool:
        return any(a <= pos < b for a, b in cell_spans)

    defined: set[str] = set()
    issues: List[InlineOrderIssue] = []
    for pos, kind, payload in sorted(events, key=lambda e: e[0]):
        if kind == "cell":
            body = payload
            defined.update(CLASSDEF.findall(body))
            defined.update(ASSIGN.findall(body))
            continue
        name, raw = payload
        if inside_cell(pos):
            continue
        if name not in defined:
            line = text.count("\n", 0, pos) + 1
            issues.append(
                InlineOrderIssue(
                    path=path,
                    line=line,
                    name=name,
                    raw=raw,
                    message=f"`{name}` referenced inline before its defining Python cell ({raw})",
                )
            )
    return issues
