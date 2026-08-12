#!/usr/bin/env python3
"""
prototype_typed_fmt.py — Proof-of-concept for the typed, *guarded* formatter
family. NOT wired into the book yet; this is the demonstration that the design
(a) makes illegal values impossible to render and (b) is mechanically
migratable. Run directly:  python3 book/tools/audit/fmt/prototype_typed_fmt.py
"""
from __future__ import annotations
import ast


class MarkdownStr(str):
    def _repr_markdown_(self):
        return str.__str__(self)


def _num(x):
    return float(x)


# ---------------------------------------------------------------------------
# Guarded typed formatters
# ---------------------------------------------------------------------------
def fmt_percent(ratio, *, precision=1, style="prose", max_ratio=1.5):
    """Format a 0-1 RATIO as a percentage. Single canonical domain.

    style="prose"  -> "85 percent"   (MIT body style, default)
    style="symbol" -> "85%"          (tables, equations, captions)

    Guard: a value outside [0, max_ratio] is almost certainly an
    already-scaled (0-100) value passed by mistake -> raise, never render.
    This is the structural fix for the 'no 10,000%' requirement.
    """
    r = _num(ratio)
    if not (0.0 <= r <= max_ratio):
        raise ValueError(
            f"fmt_percent expects a 0-1 ratio, got {r}. "
            f"If this is an already-scaled 0-100 value, divide by 100 at the "
            f"source. If a >100% value is intended (e.g. growth), pass "
            f"max_ratio explicitly. This guard prevents silent 100x errors "
            f"like 0.85 -> '8500 percent'."
        )
    val = r * 100
    body = f"{val:.{precision}f}"
    glyph = " percent" if style == "prose" else "%"
    return MarkdownStr(f"{body}{glyph}")


def fmt_pp(points, *, precision=1, style="prose"):
    """Percentage-POINTS (a difference of two percentages), already 0-100 scale."""
    v = _num(points)
    body = f"{v:.{precision}f}"
    glyph = " percentage points" if style == "prose" else " pp"
    return MarkdownStr(f"{body}{glyph}")


def fmt_multiple(x, *, precision=1, commas=False):
    """A multiplier/factor as a NUMBER ONLY. Prose supplies $\\times$.
    Guard: multipliers are non-negative."""
    v = _num(x)
    if v < 0:
        raise ValueError(f"fmt_multiple expects a non-negative factor, got {v}.")
    sep = "," if commas else ""
    body = f"{v:{sep}.{precision}f}"
    return MarkdownStr(body)


# ---------------------------------------------------------------------------
# Demonstration 1: the guard makes 10,000% impossible to render
# ---------------------------------------------------------------------------
def demo_guard():
    print("== Demo 1: guard behavior ==")
    cases = [
        ("fmt_percent(0.85)", lambda: fmt_percent(0.85)),
        ("fmt_percent(0.85, style='symbol')", lambda: fmt_percent(0.85, style="symbol")),
        ("fmt_percent(0.123, precision=1)", lambda: fmt_percent(0.123, precision=1)),
        ("fmt_percent(85)  # the 10,000% bug", lambda: fmt_percent(85)),
        ("fmt_percent(1.20, max_ratio=2)  # explicit >100%", lambda: fmt_percent(1.20, max_ratio=2)),
        ("fmt_multiple(-3)  # impossible factor", lambda: fmt_multiple(-3)),
        ("fmt_pp(7.0)", lambda: fmt_pp(7.0)),
    ]
    for label, fn in cases:
        try:
            print(f"  OK    {label:42s} -> {fn()!r}")
        except ValueError as e:
            print(f"  RAISE {label:42s} -> {str(e).splitlines()[0]}")


# ---------------------------------------------------------------------------
# Demonstration 2: AST codemod transforms real call patterns precisely
# ---------------------------------------------------------------------------
SAMPLES = [
    # (source line, the human-verified intended rewrite)
    "x_str = fmt(robust_acc * 100, precision=0, commas=False, suffix='%')",
    "y_str = fmt(u_mfu * 100, suffix=' percent')",
    "z_str = fmt(small_efficiency_pct, suffix=' percent')",
    "m_str = fmt(speedup, precision=1, commas=False, suffix='×')",
]


class PercentRewriter(ast.NodeTransformer):
    """Rewrite fmt(<expr>*100, suffix='%'/' percent') -> fmt_percent(<expr>,...).

    Only fires when the argument is literally `<expr> * 100` (provably a ratio
    *100), so it never mis-handles the already-scaled `_pct` case, which is
    flagged for human review instead."""
    def __init__(self):
        self.notes = []

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in {"fmt", "fmt_int"}):
            return node
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        suf = kw.get("suffix")
        if not (isinstance(suf, ast.Constant) and isinstance(suf.value, str)):
            return node
        sval = suf.value.strip()
        if sval not in {"%", "percent"}:
            return node
        if not node.args:
            return node
        arg = node.args[0]
        # SAFE case: argument is `<expr> * 100` -> unwrap to the ratio
        if (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult)
                and isinstance(arg.right, ast.Constant) and arg.right.value == 100):
            ratio = arg.left
            style = ast.Constant("symbol" if sval == "%" else "prose")
            prec = kw.get("precision", ast.Constant(1))
            new = ast.Call(
                func=ast.Name(id="fmt_percent", ctx=ast.Load()),
                args=[ratio],
                keywords=[ast.keyword(arg="precision", value=prec),
                          ast.keyword(arg="style", value=style)],
            )
            self.notes.append("auto-rewrote (ratio*100)")
            return ast.copy_location(new, node)
        # UNSAFE/ambiguous: already-scaled value -> leave, flag for human
        self.notes.append("FLAGGED for human review (already-0-100 value)")
        return node


def demo_codemod():
    print("\n== Demo 2: AST codemod (precise, only rewrites provable ratios) ==")
    for src in SAMPLES:
        tree = ast.parse(src)
        rw = PercentRewriter()
        new_tree = rw.fix_missing_locations(rw.visit(tree)) if hasattr(rw, "fix_missing_locations") else ast.fix_missing_locations(rw.visit(tree))
        out = ast.unparse(new_tree)
        note = rw.notes[-1] if rw.notes else "unchanged"
        print(f"  IN : {src}")
        print(f"  OUT: {out}")
        print(f"       [{note}]\n")


if __name__ == "__main__":
    demo_guard()
    demo_codemod()
