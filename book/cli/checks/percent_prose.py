#!/usr/bin/env python3
"""Percent-in-prose: detect the `%` symbol where body prose should spell out 'percent'.

This is the symmetric inverse of ``percent_tables.py``. House style (MIT Press
``AU_QUERY_RESPONSES.md`` Category H, aligned with Chicago Manual of Style
§9.18) splits on *sentence vs. data*, not on context type:

  - Running prose — body text, caption sentences, alt text — spells out the
    word ``percent`` (MIT Press converted 1,275 `%` symbols in body prose).
  - Tabular / data / formula contexts — tables, equations, code, and labels
    baked into figure images — use the ``%`` symbol.

This module enforces the body-prose side: it flags a ``%`` that follows a
number in running prose. It is imported by the ``percent-in-prose`` validator
scope in ``book/cli/commands/validate.py``.

It reuses the pipe-table walk from ``percent_tables.py`` to *exclude* table
rows (where ``%`` is correct), and additionally exempts the legitimate
non-prose contexts where ``%`` is fine or must be left verbatim:

  - fenced code blocks and inline code spans
  - inline/display math (``$…$`` / ``$$…$$``)
  - Quarto attribute braces (``{width=100%}``, ``{#fig-…}``)
  - markdown link / image targets (URLs, paths — ``…%20…``)
  - raw HTML tags and ``<style>…</style>`` CSS blocks (``width: 100%;``)
  - HTML comments
  - figure/table caption + alt-text attribute lines (owned by the caption
    rule, which keeps the spelled-out word)
  - cell-option lines (``#|``) and caption lines (leading ``:``)
  - **quoted material** — a ``%`` inside double quotes is a verbatim
    quotation (Chicago: never alter a quotation), e.g. a tweet reading
    ``"100% NOT OK"``.

Ranges (``30–50%``) and other prose percentages are fixed by hand because the
prose form needs judgment (``30 to 50 percent``); there is intentionally no
auto-fixer here — only the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from cli.checks.percent_tables import _table_row_indices

# A number immediately before % in (masked) prose.
PCT_PROSE_PAT = re.compile(r"\d[\d.,]*\s*%")

CODE_FENCE_RE = re.compile(r"^\s*```")
_CAPTION_ATTRS = ("fig-cap=", "tbl-cap=", "fig-alt=", "tbl-alt=", "alt=", "title=")


@dataclass(frozen=True)
class Hit:
    line: int            # 1-based line number
    match: str           # the matched text, e.g. "98.5%"
    context: str         # trimmed surrounding text (from the masked line)


def _mask_nonprose(line: str) -> str:
    """Blank out spans where a ``%`` is legitimate or must stay verbatim.

    Order matters: code/math first (they may contain braces/quotes), then
    attribute braces, link targets, raw HTML, and finally quotations.
    """
    line = re.sub(r"`[^`]*`", "", line)            # inline code
    line = re.sub(r"\$\$?[^$]*\$\$?", "", line)    # inline / display math
    line = re.sub(r"\{[^}]*\}", "", line)          # {width=100%}, {#fig-…}
    line = re.sub(r"\]\([^)]*\)", "]()", line)     # md link / image targets
    line = re.sub(r"<[^>]*>", "", line)            # html tags / autolinks
    line = re.sub(r'"[^"]*"', "", line)            # straight double quotes
    line = re.sub(r"“[^”]*”", "", line)  # curly double quotes
    return line


def find_prose_percent(text: str) -> List[Hit]:
    """Find every number-followed-by-``%`` in running body prose in *text*."""
    lines = text.splitlines()
    table_rows = set(_table_row_indices(lines))

    hits: List[Hit] = []
    in_code = False
    in_html_comment = False
    in_style = False
    for idx, line in enumerate(lines):
        s = line.strip()

        if in_html_comment:
            if "-->" in s:
                in_html_comment = False
            continue
        if s.startswith("<!--") and "-->" not in s:
            in_html_comment = True
            continue

        if in_style:
            if "</style>" in s.lower():
                in_style = False
            continue
        if "<style" in s.lower() and "</style>" not in s.lower():
            in_style = True
            continue

        if CODE_FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code or idx in table_rows:
            continue
        if s.startswith("#|") or s.startswith(":"):
            continue
        if any(attr in line for attr in _CAPTION_ATTRS):
            continue

        masked = _mask_nonprose(line)
        for m in PCT_PROSE_PAT.finditer(masked):
            ctx = masked[max(0, m.start() - 25): m.end() + 8].strip()
            hits.append(Hit(line=idx + 1, match=m.group().strip(), context=ctx))
    return hits


def audit(paths) -> List[tuple]:
    """Standalone helper: return (file, line, match) tuples for *paths*."""
    out: List[tuple] = []
    for raw in paths:
        p = Path(raw)
        files = p.rglob("*.qmd") if p.is_dir() else ([p] if p.suffix == ".qmd" else [])
        for f in files:
            for hit in find_prose_percent(f.read_text(encoding="utf-8", errors="replace")):
                out.append((str(f), hit.line, hit.match))
    return out
