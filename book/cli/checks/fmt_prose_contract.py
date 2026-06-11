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
  * ``fmt_multiple`` / ``fmt_multiple_range`` — string owns the visible ``×`` glyph;
    prose must not repeat it, and the export name must contain a canonical
    ``mult`` token before ``_str``.

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

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
FENCE_RE = re.compile(r"^([ \t]*)```+\s*\{python\}\s*$")
CLOSE_RE = re.compile(r"^([ \t]*)```+\s*$")

# how much prose to look at on each side of a ref
WIN = 28


@dataclass
class Violation:
    code: str
    file: str
    line: int
    ref: str
    msg: str


def extract_python_cells(text: str):
    """Yield ``(fence_line, source)`` for each `````{python}``` fenced block."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = FENCE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fence_line = i + 1
        indent = m.group(1)
        body: list[str] = []
        j = i + 1
        while j < n:
            cm = CLOSE_RE.match(lines[j])
            if cm and len(cm.group(1)) <= len(indent):
                break
            line = lines[j]
            if indent and line.startswith(indent):
                line = line[len(indent):]
            body.append(line)
            j += 1
        yield fence_line, "\n".join(body)
        i = j + 1


def _const_str(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _has_mult_token(name: str) -> bool:
    if not name.endswith("_str"):
        return False
    return "mult" in name[:-4].split("_")


def _target_names(node: ast.AST) -> list[str]:
    targets = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    out = []
    for target in targets:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Attribute):
            out.append(target.attr)
    return out


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    return fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)


def _kwargs(call: ast.Call) -> dict:
    kwargs = {kw.arg: _const_str(kw.value) for kw in call.keywords if kw.arg}
    if any(kw.arg == "scale" for kw in call.keywords):
        kwargs["__has_scale__"] = "1"
    return kwargs


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
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        fname = _call_name(call)
        if not fname or not fname.startswith("fmt"):
            continue
        kwargs = _kwargs(call)
        for name in _target_names(node):
            if name and name.endswith(("_str", "_math", "_eq", "_frac")):
                out[_qual(node.lineno) + name] = (fname, kwargs)
    return out


def build_formatter_records(text: str) -> dict[str, tuple[str, dict, int, str]]:
    """{qualified_name -> (formatter, kwargs, file_line, bare_name)}."""
    out: dict[str, tuple[str, dict, int, str]] = {}
    for fence_line, source in extract_python_cells(text):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        classes = [
            (n.name, n.lineno, n.end_lineno or n.lineno)
            for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        ]

        def _qual(lineno: int) -> str:
            best = None
            for nm, s, e in classes:
                if s <= lineno <= e and (best is None or s > best[1]):
                    best = (nm, s, e)
            return best[0] + "." if best else ""

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fname = _call_name(call)
            if not fname or not fname.startswith("fmt"):
                continue
            kwargs = _kwargs(call)
            file_line = fence_line + (node.lineno or 0)
            for name in _target_names(node):
                if name and name.endswith(("_str", "_math", "_eq", "_frac")):
                    out[_qual(node.lineno) + name] = (fname, kwargs, file_line, name)
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
_COMPACT_MULT_AFTER_RE = re.compile(r"^(\$\\times\$|\\times|×|x\b)")

# The "N× <device>" hardware-count idiom (e.g. "8× H100", "4× A100 node",
# "`{python} node_gpus_str`× H100 GPUs") is a count of accelerators, not a
# multiplier: prose supplies × as shorthand for "N of". A GPU count must NOT be
# rebuilt via fmt_multiple. This is established book-wide convention (see
# compute_infrastructure, performance_engineering, inference), so a generic
# fmt/fmt_int export used this way is correct and is not flagged.
_HW_AFTER_MULT_RE = re.compile(
    r"^(?:\$\\times\$|\\times|×|x\b) *"
    r"(?:H100|H200|H800|A100|A800|A10|A30|A40|V100|B100|B200|GB200|GH200|"
    r"L4|L40S?|T4|MI\d{3}X?|TPU\w*|GPUs?|accelerators?)\b",
    re.IGNORECASE,
)


def check_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records = build_formatter_records(text)
    fmap = {name: (fname, kwargs) for name, (fname, kwargs, _, _) in records.items()}
    rel = str(path)
    out: list[Violation] = []

    for qualified, (fname, _, file_line, bare_name) in records.items():
        if fname in {"fmt_multiple", "fmt_multiple_range"} and not _has_mult_token(bare_name):
            out.append(Violation("mult_suffix", rel, file_line, qualified,
                f"{fname} now owns the times glyph; name the export with a canonical "
                f"'mult' token before _str (for example, *_mult_str)."))

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

            if (fname in {"fmt", "fmt_int"} and _COMPACT_MULT_AFTER_RE.match(after)
                    and not _HW_AFTER_MULT_RE.match(after)):
                out.append(Violation("mult_wrong_formatter", rel, lineno, ref,
                    f"{fname} is a generic number formatter, but prose uses it "
                    f"as a compact multiplier with a times glyph. Use fmt_multiple "
                    f"for the export, or add spaces around the glyph if this is "
                    f"an arithmetic product."))
            elif fname in {"fmt_percent", "fmt_percent_range"}:
                style = kwargs.get("style") or "number"
                if style in ("prose", "symbol") and _PERCENT_RE.match(after):
                    out.append(Violation("percent_dup", rel, lineno, ref,
                        f"{fname}(style={style!r}) already carries the glyph; "
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
            elif fname in {"fmt_multiple", "fmt_multiple_range"}:
                mm = _MULT_AFTER_RE.match(after)
                if mm:
                    out.append(Violation("mult_double_glyph", rel, lineno, ref,
                        f"{fname} already carries the times glyph; prose repeats a "
                        f"times glyph after the ref: …{after.strip()[:12]!r}"))
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
