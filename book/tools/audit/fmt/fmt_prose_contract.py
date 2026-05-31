#!/usr/bin/env python3
"""
fmt_prose_contract.py — enforce the OUTPUT-formatter ↔ prose glyph contract.

Static checker for invariant **I2** (ASSESSMENT.md §3 / fmt.md §7). For every
``{python} Class.var_str`` reference in prose it looks up the formatter that
produced ``var_str`` (by AST-parsing the chapter's cells) and verifies the
surrounding prose respects who owns the glyph:

  * ``fmt_percent(style='prose'|'symbol')`` / ``fmt_pp`` — string already carries
    "percent"/"%"/"percentage points"; prose must NOT repeat it.
    (``style='number'`` is the bare form — prose is *expected* to supply the word,
    so it is not flagged.)
  * ``fmt_usd`` — string owns ``$``; prose must not type a leading ``$``.
  * ``fmt_count(scale=...)`` — string owns the K/M/B glyph; prose must not repeat it.
  * ``fmt_multiple`` — string is a bare number; prose MUST add ``$\\times$``
    (and must not use a literal ``×`` — see math.md §6 #14).

This is the "is the output in line with how it's used in prose?" gate.
Read-only. Exit 0 = clean, 1 = violations.

Usage::

    python3 book/tools/audit/fmt/fmt_prose_contract.py book/quarto/contents/vol1/training/training.qmd
    python3 book/tools/audit/fmt/fmt_prose_contract.py --root book/quarto/contents
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fmt_usage import extract_python_cells  # noqa: E402

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")

# how much prose to look at on each side of a ref
WIN = 28


@dataclass
class Violation:
    code: str
    file: str
    line: int
    ref: str
    msg: str


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def build_formatter_map(cells_text: str) -> dict:
    """{name -> (formatter, {kwarg: literal})} from `x = fmt_KIND(...)`.

    Keyed by the QUALIFIED export name — ``Class.attr`` for class-body
    assignments, bare ``attr`` for module-level — because prose refs are
    qualified (``Class.attr``) and two classes can export the same bare attr
    with *different* formatters (e.g. a plain ``fmt`` vs an ``fmt_usd``). Keying
    bare would collide them and mis-attribute the glyph rule. Last write wins.
    """
    out: dict[str, tuple[str, dict]] = {}
    try:
        tree = ast.parse(cells_text)
    except SyntaxError:
        return out
    classes = [(n.name, n.lineno, n.end_lineno or n.lineno)
               for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    def _qual(lineno: int) -> str:
        best = None
        for nm, s, e in classes:
            if s <= lineno <= e and (best is None or s > best[1]):
                best = (nm, s, e)
        return best[0] + "." if best else ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        fn = call.func
        fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
        if not fname or not fname.startswith("fmt"):
            continue
        kwargs = {kw.arg: _const_str(kw.value) for kw in call.keywords if kw.arg}
        if any(kw.arg == "scale" for kw in call.keywords):
            kwargs["__has_scale__"] = "1"
        for tgt in node.targets:
            name = None
            if isinstance(tgt, ast.Name):
                name = tgt.id
            elif isinstance(tgt, ast.Attribute):
                name = tgt.attr
            if name and name.endswith(("_str", "_math", "_eq", "_frac")):
                out[_qual(node.lineno) + name] = (fname, kwargs)
    return out


def _lookup(fmap: dict, ref: str):
    """Resolve a prose ref to its formatter entry.

    Exact qualified match first (``Class.attr``). For a bare or unresolved ref,
    fall back to entries sharing the last component only when they ALL agree on
    the formatter+kwargs — an ambiguous bare name (two classes, two formatters)
    returns None so the checker stays silent rather than flag a false positive.
    """
    if ref in fmap:
        return fmap[ref]
    bare = ref.split(".")[-1]
    cands = [v for k, v in fmap.items() if k == bare or k.endswith("." + bare)]
    if not cands:
        return None
    sig = {(f, tuple(sorted((kw or {}).items()))) for f, kw in cands}
    return cands[0] if len(sig) == 1 else None


_PERCENT_RE = re.compile(r"^\s*(%|percent|percentage)", re.IGNORECASE)
_PP_RE = re.compile(r"^\s*(percentage points?|pp)\b", re.IGNORECASE)
_SCALE_RE = re.compile(r"^\s*(K|M|B|T|million|billion|thousand|trillion)\b")
_MULT_AFTER_RE = re.compile(r"^\s*(\$\\times\$|\\times|×|x\b)")


def check_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cells = "\n".join(src for _, src in extract_python_cells(text))
    fmap = build_formatter_map(cells)
    rel = str(path)
    out: list[Violation] = []

    in_cell = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if in_cell:
            continue
        for m in INLINE_PY.finditer(line):
            ref = m.group(1)
            entry = _lookup(fmap, ref)
            if not entry:
                continue  # unknown / ambiguous / not a fmt export
            fname, kwargs = entry
            after = line[m.end():m.end() + WIN]
            # currency prefix = '$' (or '\\$') *immediately* before the opening
            # backtick — a space-separated '$' is a math delimiter (e.g. $\times$).
            prefix = line[max(0, m.start() - 2):m.start()]

            if fname == "fmt_percent":
                style = kwargs.get("style") or "number"
                if style in ("prose", "symbol") and _PERCENT_RE.match(after):
                    out.append(Violation("percent_dup", rel, lineno, ref,
                        f"fmt_percent(style={style!r}) already carries the glyph; "
                        f"prose repeats it: …{after.strip()[:20]!r}"))
            elif fname == "fmt_pp":
                if _PP_RE.match(after):
                    out.append(Violation("pp_dup", rel, lineno, ref,
                        "fmt_pp already carries 'percentage points'; prose repeats it"))
            elif fname == "fmt_usd":
                if prefix.endswith("$"):  # '$`{python}' or '\\$`{python}'
                    out.append(Violation("usd_dup", rel, lineno, ref,
                        "fmt_usd already carries '$'; prose prepends another"))
            elif fname == "fmt_count" and kwargs.get("__has_scale__"):
                if _SCALE_RE.match(after):
                    out.append(Violation("scale_dup", rel, lineno, ref,
                        f"fmt_count(scale=…) already carries the scale glyph; "
                        f"prose repeats it: …{after.strip()[:12]!r}"))
            elif fname == "fmt_multiple":
                mm = _MULT_AFTER_RE.match(after)
                if not mm:
                    out.append(Violation("mult_missing_glyph", rel, lineno, ref,
                        "fmt_multiple is a bare number; prose must add $\\times$ "
                        f"after the ref (got: …{after.strip()[:12]!r})"))
                elif "×" in mm.group(0) or mm.group(0).strip() == "x":
                    out.append(Violation("mult_literal_x", rel, lineno, ref,
                        "use $\\times$ not a literal ×/x after a fmt_multiple ref"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("qmd", nargs="*", help="chapter .qmd file(s)")
    ap.add_argument("--root", help="scan all .qmd under this dir")
    args = ap.parse_args()

    files: list[Path] = [Path(p) for p in args.qmd]
    if args.root:
        files += sorted(Path(args.root).rglob("*.qmd"))
    if not files:
        ap.error("pass .qmd file(s) or --root")

    violations: list[Violation] = []
    for f in files:
        if f.is_file():
            violations.extend(check_file(f))

    for v in violations:
        short = v.file.split("contents/")[-1]
        print(f"{short}:{v.line}  [{v.code}]  {{python}} {v.ref}\n    {v.msg}")
    print(f"\n{len(violations)} prose-contract violation(s) across {len(files)} file(s).")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
