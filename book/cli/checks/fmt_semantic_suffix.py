#!/usr/bin/env python3
"""
fmt_semantic_suffix check — value-kind must be a typed formatter, not a
free-text ``suffix=`` on the generic ``fmt()``/``fmt_int()``.

This is the regression gate for the typed-formatter migration. The semantic
kind of a value (percent, multiplier, percentage-points, count-scale) belongs
in the *function name* — ``fmt_percent``, ``fmt_multiple``, ``fmt_pp``,
``fmt_count`` — each of which carries a domain guard (e.g. ``fmt_percent``
cannot emit 10,000%). Encoding the kind as a ``suffix=`` string defeats that
guard and was the source of the ratio-vs-already-scaled ambiguity.

Reserved: ``suffix=`` on ``fmt()``/``fmt_int()`` for honest *physical-unit*
labels (`` " ms"``, `` " GB/s"``) only. Currency (``fmt_usd``) and physical
quantities (``fmt_qty``) have their own helpers and are not inspected here.

Invoked by::

    ./book/binder check math --scope suffix-semantics

Opt-in (``default=False``) until the corpus migration completes; flip the
Scope to ``default=True`` to make it a pre-commit gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FENCE_OPEN = re.compile(r"^([ \t]*)```+\s*\{python\}\s*$")
FENCE_CLOSE = re.compile(r"^([ \t]*)```+\s*$")

# Only these two generic formatters are inspected. fmt_usd / fmt_percent /
# fmt_qty / fmt_pp / fmt_count / fmt_multiple legitimately own their own
# suffix or glyph semantics.
INSPECTED_FUNCS = {"fmt", "fmt_int"}

PERCENT_VALUES = {
    "%", "percent",
}
PP_VALUES = {
    "percentage points", "percentage point", "pp",
}
SCALE_VALUES = {"K", "M", "B", "T"}
RATE_VALUES = {
    "QPS", "FPS", "tokens/s", "img/s", "images/s", "req/s", "samples/s",
}


@dataclass
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str
    suggestion: str = ""


def _iter_python_cells(text: str):
    """Yield (fence_line_1based, dedented_source) for each ```{python}``` block."""
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        m = FENCE_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        indent = m.group(1)
        fence_line = i + 1
        body = []
        j = i + 1
        while j < n:
            if FENCE_CLOSE.match(lines[j]):
                break
            ln = lines[j]
            if indent and ln.startswith(indent):
                ln = ln[len(indent):]
            body.append(ln)
            j += 1
        yield fence_line, "\n".join(body)
        i = j + 1


def _literal_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _classify(suffix: str):
    """Return (code, suggestion) if the suffix encodes a semantic kind, else None."""
    stripped = suffix.strip()
    if stripped in PERCENT_VALUES:
        return (
            "percent_in_suffix",
            "Use fmt_percent(ratio, style='prose'|'symbol'); pass the 0-1 "
            "ratio (divide an already-0-100 value by 100 at the source).",
        )
    if stripped in PP_VALUES:
        return (
            "pp_in_suffix",
            "Use fmt_pp(points, style='prose'|'symbol') — percentage points "
            "are a 0-100 difference, distinct from a percent share.",
        )
    if stripped in SCALE_VALUES:
        return (
            "scale_glyph_in_suffix",
            f"Use fmt_count(value, scale='{stripped}') so the magnitude scale "
            "and glyph are declared together. (Currency uses fmt_usd.)",
        )
    if stripped in RATE_VALUES:
        return (
            "rate_in_suffix",
            f"Use fmt_rate(value, '{stripped}') so service-rate labels are "
            "allowlisted and checked by the formatter.",
        )
    if stripped == "x" or "×" in suffix:
        return (
            "multiplier_in_suffix",
            "Use fmt_multiple(x) for the number and write the glyph in prose "
            "as $\\times$ (see .claude/rules/math.md §6 #14).",
        )
    return None


def _audit_file(qmd_path: Path) -> list[Violation]:
    out: list[Violation] = []
    rel = str(qmd_path)
    text = qmd_path.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()
    for fence_line, cell in _iter_python_cells(text):
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None)
            if fname not in INSPECTED_FUNCS:
                continue
            for kw in node.keywords:
                if kw.arg != "suffix":
                    continue
                literal = _literal_str(kw.value)
                if literal is None:
                    continue
                hit = _classify(literal)
                if hit is None:
                    continue
                code, suggestion = hit
                file_line = fence_line + (node.lineno or 0)
                ctx = ""
                if 0 < file_line <= len(raw_lines):
                    ctx = raw_lines[file_line - 1].strip()[:160]
                out.append(Violation(
                    file=rel,
                    line=file_line,
                    code=code,
                    message=(
                        f"{fname}(..., suffix={literal!r}) encodes value-kind "
                        f"in a free-text suffix. The kind belongs in the "
                        f"formatter name so its domain guard applies."
                    ),
                    context=ctx,
                    suggestion=suggestion,
                ))
    return out


def audit(paths: list[Path]) -> list[Violation]:
    all_v: list[Violation] = []
    for p in paths:
        files = [p] if p.is_file() else sorted(p.rglob("*.qmd"))
        for f in files:
            all_v.extend(_audit_file(f))
    return all_v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    targets = args.paths or [Path("book/quarto/contents")]
    violations = audit(targets)
    if args.json:
        print(json.dumps([v.__dict__ for v in violations], indent=2))
        return 1 if violations else 0
    for v in violations:
        print(f"{v.file}:{v.line}  [{v.code}]  {v.message}")
        if v.suggestion:
            print(f"  -> {v.suggestion}")
        print(f"  {v.context}\n")
    print(f"Total violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
