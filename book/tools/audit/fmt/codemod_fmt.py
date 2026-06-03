#!/usr/bin/env python3
"""
codemod_fmt.py — paired (cell + prose) rewrites for the typed-fmt migration.

Two strictly separated lanes (PLAN.md §6 WS1, ASSESSMENT.md §0a):

RETIRED (pre-Design-B) — the old multiplier relocation lane
    cell:  ``X = fmt(f, ...,  suffix='×')``  →  ``X = fmt_multiple(f, ...)``
    prose: ``…`{python} C.X`…``              →  ``…`{python} C.X`$\\times$…``
  This lane is obsolete. Design B makes ``fmt_multiple`` own ``$\\times$`` and
  requires ``*_mult_str`` export names. The public ``multiple`` command now
  refuses to run so it cannot reintroduce prose-owned multiplier glyphs.

AMBIGUOUS (never auto-touched) — emitted to an adjudication queue
    * suffix='%'  — fmt_percent needs a 0-1 *ratio*; whether the source value is
      a ratio or already-scaled 0-100 is NOT statically knowable. Human decides
      the source scale + MIT style (body 'percent' vs table '%').
    * suffix in {K,M,B,T,…} — fmt_count needs raw value + scale; the division
      may already be in the expression. Human decides.
    * multi-line multiplier calls — surgical single-line replace can't prove them.

DEFAULT IS DRY-RUN. Nothing is written without ``--write``. After ``--write``,
run the gauntlet: assess_equiv (value+visible-prose diff) then
fmt_prose_contract, then the render checks (MIGRATION.md Phase 3).

Usage::

    # emit the human-adjudication queue (percent / scale / multi-line)
    python3 codemod_fmt.py queue --root book/quarto/contents --out /tmp/fmt_queue.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fmt_usage import extract_python_cells  # noqa: E402

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")

MULT_GLYPHS = {"×", "x", " ×", "× "}
PERCENT_GLYPHS = {"%", " %", "%%"}
SCALE_GLYPHS = {"K", "M", "B", "T", "k", "G"}
REWRITABLE_FNS = {"fmt", "fmt_int"}

_TIMES_AFTER = re.compile(r"^\s*(\$\\times\$|\\times|×)")


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _assign_target(node: ast.Assign):
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            return tgt.id
        if isinstance(tgt, ast.Attribute):
            return tgt.attr
    return None


@dataclass
class MultEdit:
    line: int          # 1-based file line of the call
    var: str
    old: str           # exact call source segment
    new: str           # rewritten call


@dataclass
class QueueItem:
    file: str
    line: int
    var: str | None
    fn: str
    suffix: str
    kind: str          # 'percent' | 'scale' | 'multiline-multiplier' | 'other'
    call: str
    action: str        # the human decision needed


# percent suffix -> fmt_percent style that reproduces the SAME glyph/word.
# (exact forms only; spacing variants like ' %' / 'percent' are queued)
PERCENT_STYLE = {"%": "symbol", " percent": "prose"}

_NEEDS_PAREN = (ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp, ast.IfExp, ast.Lambda)


def _is_const_100(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == 100 and not isinstance(node.value, bool)


def _ratio_source(arg, src: str) -> str | None:
    """Source for the 0-1 ratio fmt_percent expects, given the already-scaled
    fmt argument.

    * ``ratio * 100`` / ``100 * ratio``  -> strip the scaling: pass ``ratio``.
      The cleanest, most honest result (``fmt_percent(utilization, …)``), and
      still byte-identical (fmt_percent multiplies it back by 100).
    * anything else (a standalone ``foo_pct`` number, a literal, a call) ->
      divide by 100, parenthesised only when precedence requires it.
    """
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
        inner = None
        if _is_const_100(arg.right):
            inner = arg.left
        elif _is_const_100(arg.left):
            inner = arg.right
        if inner is not None:
            return ast.get_source_segment(src, inner)
    seg = ast.get_source_segment(src, arg)
    if seg is None:
        return None
    return f"({seg})/100" if isinstance(arg, _NEEDS_PAREN) else f"{seg}/100"


def _rewrite_percent(call, src: str) -> str | None:
    """fmt(x, …, suffix='%'|' percent') -> fmt_percent(<ratio>, …, style=…).

    Byte-identical by construction: the 0-1 ratio fmt_percent expects renders x
    back with the same glyph ('%'->symbol) or word (' percent'->prose). Moves
    percent into the guarded formatter (no 10,000% possible) with no change to
    rendered output. Values above 150% trip fmt_percent's max_ratio guard ->
    exec fails -> the byte-identical gate reverts and the site is queued.
    """
    seg = ast.get_source_segment(src, call) or ""
    if "prefix=" in seg or not call.args:
        return None
    suffix = None
    for kw in call.keywords:
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
    if suffix not in PERCENT_STYLE:
        return None
    ratio = _ratio_source(call.args[0], src)
    if ratio is None or "\n" in ratio:
        return None
    extra = ""
    for kw in call.keywords:
        if kw.arg == "precision":
            extra += f", precision={ast.get_source_segment(src, kw.value)}"
        elif kw.arg == "commas":
            extra += f", commas={ast.get_source_segment(src, kw.value)}"
    return f"fmt_percent({ratio}{extra}, style='{PERCENT_STYLE[suffix]}')"


def _rewrite_percent_int(call, src: str, style: str) -> str | None:
    """fmt_int(x, …, suffix='%'|' percent') -> fmt_percent(round(x)/100, …).

    fmt_int is fmt(round(val), precision=0, commas=True), so rounding x first
    and dividing to a ratio reproduces the integer percent byte-for-byte
    (round(x)/100 * 100 == round(x) exactly for the integers fmt_int yields).
    """
    seg = ast.get_source_segment(src, call) or ""
    if "prefix=" in seg or not call.args:
        return None
    arg0 = ast.get_source_segment(src, call.args[0])
    if arg0 is None or "\n" in arg0:
        return None
    commas = "True"  # fmt_int default
    for kw in call.keywords:
        if kw.arg == "commas" and isinstance(kw.value, ast.Constant):
            commas = "True" if kw.value.value else "False"
    inner = arg0 if arg0.strip().startswith("round(") else f"round({arg0})"
    return f"fmt_percent({inner}/100, precision=0, commas={commas}, style='{style}')"


def scan_percent(path: Path):
    """Return (percent_edits, queue) for the byte-identical percent lane."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in ("fmt", "fmt_int"):
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None:
                continue
            sfx = suffix.strip().lower()
            if sfx not in ("%", "percent", "percentage", "%%"):
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            if fname == "fmt":
                new = _rewrite_percent(call, src)
            elif fname == "fmt_int" and suffix in PERCENT_STYLE:
                new = _rewrite_percent_int(call, src, PERCENT_STYLE[suffix])
            else:
                new = None
            # a multiline call can't be spliced by the single-line applier — route
            # it to the queue rather than emit an edit that silently won't apply.
            if new and var and "\n" not in seg:
                edits.append(MultEdit(file_line, var, seg, new))
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "percent", seg.replace("\n", " ⏎ "),
                    "non-exact percent suffix / fmt_int / prefix / multiline: convert "
                    "by hand to fmt_percent((x)/100, style='symbol'|'prose') by context."))
    return edits, queue


# scale glyph -> magnitude, and the named/literal divisors that match each.
_SCALE_FACTOR = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_NAMED_DIVISOR = {"THOUSAND": 1e3, "MILLION": 1e6, "BILLION": 1e9, "TRILLION": 1e12}
_SCALE_FACTOR_NAME = {"K": "THOUSAND", "M": "MILLION", "B": "BILLION", "T": "TRILLION"}
_PARAM_UNIT_SCALE = {"Kparam": "K", "Mparam": "M", "Bparam": "B", "Tparam": "T"}


def _divisor_factor(node) -> float | None:
    """Magnitude of a divisor node if it's a recognised scale constant."""
    if isinstance(node, ast.Name):
        return _NAMED_DIVISOR.get(node.id)
    if isinstance(node, ast.Attribute):
        return _NAMED_DIVISOR.get(node.attr)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        v = float(node.value)
        return v if v in set(_SCALE_FACTOR.values()) else None
    return None


def _rewrite_scale(call, src: str) -> str | None:
    """fmt(x / MILLION, …, suffix='M') -> fmt_count(x, scale='M', …).

    Only the clean, byte-identical case: the argument is ``<x> / <divisor>``
    where the divisor's magnitude matches the (exact, uppercase) glyph, so
    stripping the division and declaring scale= reproduces the rendered value
    exactly (fmt_count divides by the same factor). Pre-scaled values (no
    division), lowercase 'k', space-padded glyphs, fmt_int and prefix= are NOT
    handled here — they are queued for a source-level decision.
    """
    seg = ast.get_source_segment(src, call) or ""
    if "prefix=" in seg or not call.args:
        return None
    suffix = None
    for kw in call.keywords:
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
    if suffix not in _SCALE_FACTOR:  # exact uppercase glyph, no spaces
        return None
    arg = call.args[0]
    if not (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div)):
        return None
    if _divisor_factor(arg.right) != _SCALE_FACTOR[suffix]:
        return None  # divisor must match the glyph (else a latent bug — leave it)
    raw = ast.get_source_segment(src, arg.left)
    if raw is None or "\n" in raw:
        return None
    precision = "1"  # fmt default; fmt_count default is 0, so always emit it
    commas = None
    for kw in call.keywords:
        if kw.arg == "precision":
            precision = ast.get_source_segment(src, kw.value)
        elif kw.arg == "commas":
            commas = ast.get_source_segment(src, kw.value)
    tail = f", commas={commas}" if commas is not None else ""
    return f"fmt_count({raw}, scale='{suffix}', precision={precision}{tail})"


def scan_scale(path: Path):
    """Return (scale_edits, queue) for the byte-identical scale lane."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in ("fmt", "fmt_int"):
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None or suffix.strip() not in {"K", "M", "B", "T", "k", "G"}:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            new = _rewrite_scale(call, src) if fname == "fmt" else None
            if new and var and "\n" not in seg:
                edits.append(MultEdit(file_line, var, seg, new))
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "scale", seg.replace("\n", " ⏎ "),
                    "pre-scaled value / lowercase 'k' / spaced glyph / fmt_int / "
                    "multiline: keep the RAW count at the source and use "
                    "fmt_count(raw, scale='K'|'M'|'B'|'T'); decide by hand."))
    return edits, queue


def _round_divisor_raw(node: ast.AST, scale: str, src: str) -> str | None:
    """Return raw numerator for round(x / SCALE), if it matches scale."""
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Name) and node.func.id == "round"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    inner = node.args[0]
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div)):
        return None
    if _divisor_factor(inner.right) != _SCALE_FACTOR[scale]:
        return None
    raw = ast.get_source_segment(src, inner.left)
    return raw if raw and "\n" not in raw else None


def _division_raw(node: ast.AST, scale: str, src: str) -> str | None:
    """Return raw numerator for x / SCALE or x // SCALE, if scale matches."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv))):
        return None
    if _divisor_factor(node.right) != _SCALE_FACTOR[scale]:
        return None
    raw = ast.get_source_segment(src, node.left)
    return raw if raw and "\n" not in raw else None


def _param_m_as_raw(node: ast.AST, scale: str, src: str) -> str | None:
    """Return raw parameter count for q.m_as(Kparam/Mparam/Bparam/Tparam)."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "m_as"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    unit = ast.get_source_segment(src, node.args[0])
    if unit not in _PARAM_UNIT_SCALE or _PARAM_UNIT_SCALE[unit] != scale:
        return None
    quantity = ast.get_source_segment(src, func.value)
    if not quantity or "\n" in quantity:
        return None
    if not re.match(r"^[A-Za-z_][\w.]*$", quantity):
        quantity = f"({quantity})"
    return f"{quantity}.m_as('param')"


def _prescaled_raw(node: ast.AST, scale: str, src: str) -> str | None:
    """Treat an already-scaled value as raw count by multiplying the factor back."""
    arg = ast.get_source_segment(src, node)
    if not arg or "\n" in arg:
        return None
    factor = _SCALE_FACTOR_NAME[scale]
    if re.match(r"^[A-Za-z_][\w.]*$", arg) or re.match(r"^[0-9][0-9_]*(?:\.[0-9_]+)?$", arg):
        return f"{arg} * {factor}"
    return f"({arg}) * {factor}"


def _scale_style_raw(node: ast.AST, scale: str, src: str) -> str | None:
    return (
        _division_raw(node, scale, src)
        or _round_divisor_raw(node, scale, src)
        or _param_m_as_raw(node, scale, src)
        or _prescaled_raw(node, scale, src)
    )


def _rewrite_scale_style(call: ast.Call, src: str) -> str | None:
    """Rewrite deferred scale suffixes to no-space ``fmt_count`` style.

    This is an intentional house-style normalization, not a byte-identical
    rewrite: ``"70 B"`` becomes ``"70B"`` and lowercase ``"100k"`` becomes
    ``"100K"``.
    """
    seg = ast.get_source_segment(src, call) or ""
    if "prefix=" in seg or not call.args or "\n" in seg:
        return None
    fname = call.func.id if isinstance(call.func, ast.Name) else (
        call.func.attr if isinstance(call.func, ast.Attribute) else None
    )
    if fname not in {"fmt", "fmt_int"}:
        return None
    suffix = None
    for kw in call.keywords:
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
    scale = suffix.strip().upper() if suffix else None
    if scale not in _SCALE_FACTOR:
        return None

    raw = _scale_style_raw(call.args[0], scale, src)
    if raw is None:
        return None

    precision = "0" if fname == "fmt_int" else "1"
    commas = None
    for kw in call.keywords:
        if kw.arg == "precision":
            precision = ast.get_source_segment(src, kw.value)
        elif kw.arg == "commas":
            commas = ast.get_source_segment(src, kw.value)
        elif kw.arg not in {"suffix"}:
            return None
    if fname == "fmt_int":
        arg_src = ast.get_source_segment(src, call.args[0])
        if not arg_src or "\n" in arg_src:
            return None
        factor = _SCALE_FACTOR_NAME[scale]
        scaled = arg_src if arg_src.strip().startswith("round(") else f"round({arg_src})"
        raw = f"{scaled} * {factor}"
    tail = f", commas={commas}" if commas is not None else ""
    return f"fmt_count({raw}, scale='{scale}', precision={precision}{tail})"


def scan_scale_style(path: Path):
    """Return (scale_edits, qualified_vars, queue) for the approved no-space style."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    qualified_vars: set[str] = set()
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        classes = [(n.name, n.lineno, n.end_lineno or n.lineno)
                   for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        def _qualifier(lineno: int) -> str:
            best = None
            for name, s, e in classes:
                if s <= lineno <= e and (best is None or s > best[1]):
                    best = (name, s, e)
            return best[0] + "." if best else ""

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in ("fmt", "fmt_int"):
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None or suffix.strip().upper() not in _SCALE_FACTOR:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            new = _rewrite_scale_style(call, src)
            if new and var:
                edits.append(MultEdit(file_line, var, seg, new))
                qualified_vars.add(_qualifier(node.lineno) + var)
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "scale", seg.replace("\n", " ⏎ "),
                    "Could not derive raw count for fmt_count(raw, scale=…); decide by hand."))
    return edits, qualified_vars, queue


_UNIT_BASE_SUFFIXES = (
    " GB/s", " MB/s", " KB/s", " TB/s",
    " TFLOP/s", " GFLOP/s", " PFLOP/s", " EFLOP/s", " ZFLOP/s",
    " GB", " MB", " KB", " TB", " PB", " KiB", " MiB", " GiB", " TiB",
    " GFLOP", " TFLOP", " PFLOP", " EFLOP", " ZFLOP", " TOPS",
    " W", " mW", " kW", " Wh", " kWh",
    " ms", " µs", " ns", " min", " h",
    " mJ", " J", " mJ/MB",
)


def _unit_suffix_parts(suffix: str) -> tuple[str, str] | None:
    """Split a physical-unit suffix into (canonical base, extra suffix).

    The base must be a unit label that ``fmt_qty`` can own. A trailing
    ``/token``-style qualifier is carried as ``extra_suffix``; arbitrary words
    such as ``" hours"`` are deliberately excluded because canonical
    abbreviation would change rendered prose.
    """
    for base in sorted(_UNIT_BASE_SUFFIXES, key=len, reverse=True):
        if suffix == base:
            return base, ""
        if suffix.startswith(base + "/"):
            return base, suffix[len(base):]
    return None


def _singular_flop_unit(unit_src: str, base_suffix: str) -> str:
    """Prefer singular FLOP unit aliases when the visible suffix is singular."""
    if "FLOP" not in base_suffix:
        return unit_src
    return (unit_src
            .replace("ZFLOPs", "ZFLOP")
            .replace("EFLOPs", "EFLOP")
            .replace("PFLOPs", "PFLOP")
            .replace("TFLOPs", "TFLOP")
            .replace("GFLOPs", "GFLOP")
            .replace("MFLOPs", "MFLOP")
            .replace("KFLOPs", "KFLOP"))


_UNIT_LITERAL_EXPR = {
    "KB": "KB", "MB": "MB", "GB": "GB", "TB": "TB", "PB": "PB",
    "KiB": "KiB", "MiB": "MiB", "GiB": "GiB", "TiB": "TiB",
    "GFLOP": "GFLOP", "GFLOPs": "GFLOP",
    "TFLOP": "TFLOP", "TFLOPs": "TFLOP",
    "PFLOP": "PFLOP", "PFLOPs": "PFLOP",
    "EFLOP": "EFLOP", "EFLOPs": "EFLOP",
    "ZFLOP": "ZFLOP", "ZFLOPs": "ZFLOP",
    "TOPS": "TOPS",
    "W": "watt", "watt": "watt",
    "mW": "milliwatt", "kW": "kilowatt",
    "ms": "ms", "millisecond": "ms",
    "us": "microsecond", "µs": "microsecond", "microsecond": "microsecond",
    "ns": "nanosecond", "nanosecond": "nanosecond",
    "mJ": "ureg.millijoule", "J": "joule", "joule": "joule",
    "Wh": "ureg.Wh", "kWh": "ureg.kWh",
}


def _unit_literal_expr(literal: str) -> str | None:
    if literal in _UNIT_LITERAL_EXPR:
        return _UNIT_LITERAL_EXPR[literal]
    for sep in ("/s", "/second"):
        if literal.endswith(sep):
            num = literal[:-len(sep)]
            if num in _UNIT_LITERAL_EXPR:
                return f"{_UNIT_LITERAL_EXPR[num]}/second"
    return None


def _m_as_quantity_parts(arg: ast.AST, src: str, base_suffix: str) -> tuple[str, str] | None:
    """Return (quantity_expr, display_unit_expr) for ``q.m_as(unit)``."""
    if not isinstance(arg, ast.Call):
        return None
    func = arg.func
    if not (isinstance(func, ast.Attribute) and func.attr == "m_as"):
        return None
    if len(arg.args) != 1 or arg.keywords:
        return None
    quantity = ast.get_source_segment(src, func.value)
    unit_node = arg.args[0]
    unit = ast.get_source_segment(src, unit_node)
    if not quantity or not unit or "\n" in quantity or "\n" in unit:
        return None
    unit_lit = _const_str(unit_node)
    if unit_lit is not None:
        unit = _unit_literal_expr(unit_lit) or unit
    unit = _singular_flop_unit(unit, base_suffix)
    return quantity, unit


def _rewrite_unit(call: ast.Call, src: str) -> str | None:
    """fmt(q.m_as(unit), ..., suffix=" unit") -> fmt_qty(q, unit, ...).

    This covers only the clean Quantity-backed shape. The acceptance gate still
    proves byte identity, so mismatched labels such as ``GiB`` displayed as
    ``GB`` are rejected automatically.
    """
    seg = ast.get_source_segment(src, call) or ""
    if not call.args or "\n" in seg:
        return None

    suffix = None
    allowed_kwargs = {"precision", "commas", "prefix", "suffix"}
    kw_src: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg not in allowed_kwargs:
            return None
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
        elif kw.arg:
            val = ast.get_source_segment(src, kw.value)
            if val is None or "\n" in val:
                return None
            kw_src[kw.arg] = val
    if suffix is None:
        return None

    parts = _unit_suffix_parts(suffix)
    if parts is None:
        return None
    base_suffix, extra_suffix = parts

    qparts = _m_as_quantity_parts(call.args[0], src, base_suffix)
    if qparts is None:
        return None
    quantity, unit = qparts

    tail = ""
    if "precision" in kw_src:
        tail += f", precision={kw_src['precision']}"
    # fmt() defaults commas=True, while fmt_qty() defaults commas=False.
    tail += f", commas={kw_src.get('commas', 'True')}"
    if "prefix" in kw_src:
        tail += f", prefix={kw_src['prefix']}"
    if extra_suffix:
        tail += f", per={extra_suffix[1:]!r}"
    return f"fmt_qty({quantity}, {unit}{tail})"


def scan_unit(path: Path):
    """Return (unit_edits, queue) for clean Quantity-backed unit suffixes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname != "fmt":
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None or _unit_suffix_parts(suffix) is None:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            new = _rewrite_unit(call, src)
            if new and var:
                edits.append(MultEdit(file_line, var, seg, new))
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "unit", seg.replace("\n", " ⏎ "),
                    "unit suffix is not the clean q.m_as(unit) shape; keep as-is "
                    "or refactor the source to carry a Pint Quantity before using fmt_qty."))
    return edits, queue


_TIME_SUFFIX_INFO = {
    " ns": ("nanosecond", "symbol"),
    " μs": ("microsecond", "symbol"),
    " µs": ("microsecond", "symbol"),
    " us": ("microsecond", "symbol"),
    " ms": ("millisecond", "symbol"),
    " microsecond": ("microsecond", "word"),
    " microseconds": ("microsecond", "word"),
    " millisecond": ("millisecond", "word"),
    " milliseconds": ("millisecond", "word"),
    " s": ("second", "symbol"),
    " min": ("minute", "symbol"),
    " h": ("hour", "symbol"),
    " second": ("second", "word"),
    " seconds": ("second", "word"),
    " minute": ("minute", "word"),
    " minutes": ("minute", "word"),
    " hour": ("hour", "word"),
    " hours": ("hour", "word"),
    " day": ("day", "word"),
    " days": ("day", "word"),
    " week": ("week", "word"),
    " weeks": ("week", "word"),
    " month": ("month", "word"),
    " months": ("month", "word"),
    " year": ("year", "word"),
    " years": ("year", "word"),
}


def _rewrite_time(call: ast.Call, src: str) -> str | None:
    """fmt/fmt_int(..., suffix=' ms'|' seconds') -> fmt_time(...).

    The source spelling uses full unit names (``"millisecond"``, ``"second"``)
    so the argument is visibly a unit, while ``style`` controls whether the
    rendered suffix is compact (``ms``) or prose (``seconds``). The
    byte-identical lane rejects any site where plural agreement or a guard
    would change output.
    """
    seg = ast.get_source_segment(src, call) or ""
    if not call.args or "\n" in seg:
        return None
    fname = call.func.id if isinstance(call.func, ast.Name) else (
        call.func.attr if isinstance(call.func, ast.Attribute) else None
    )
    if fname not in {"fmt", "fmt_int"}:
        return None

    suffix = None
    allowed_kwargs = {"precision", "commas", "suffix", "approx", "lower_bound"}
    kw_src: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg not in allowed_kwargs:
            return None
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
        elif kw.arg:
            val = ast.get_source_segment(src, kw.value)
            if val is None or "\n" in val:
                return None
            kw_src[kw.arg] = val
    if suffix not in _TIME_SUFFIX_INFO:
        return None
    unit, style = _TIME_SUFFIX_INFO[suffix]

    arg0 = ast.get_source_segment(src, call.args[0])
    if arg0 is None or "\n" in arg0:
        return None
    if fname == "fmt_int":
        arg0 = arg0 if arg0.strip().startswith("round(") else f"round({arg0})"

    tail = ""
    if fname == "fmt_int":
        tail += ", precision=0"
        tail += f", commas={kw_src.get('commas', 'True')}"
    else:
        if "precision" in kw_src:
            tail += f", precision={kw_src['precision']}"
        # fmt() defaults commas=True; fmt_time() defaults commas=False.
        if "commas" in kw_src:
            tail += f", commas={kw_src['commas']}"
        else:
            tail += ", commas=True"
    if style == "word":
        tail += ", style='word'"
    for marker in ("approx", "lower_bound"):
        if marker in kw_src:
            tail += f", {marker}={kw_src[marker]}"
    return f"fmt_time({arg0}, {unit!r}{tail})"


def scan_time(path: Path):
    """Return (time_edits, queue) for exact time suffixes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in {"fmt", "fmt_int"}:
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix not in _TIME_SUFFIX_INFO:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            new = _rewrite_time(call, src)
            if new and var:
                edits.append(MultEdit(file_line, var, seg, new))
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "time", seg.replace("\n", " ⏎ "),
                    "exact time suffix could not be rewritten mechanically; "
                    "convert by hand to fmt_time(value, unit=..., style=...)."))
    return edits, queue


_RATE_SUFFIX_INFO = {
    " QPS": "QPS",
    " FPS": "FPS",
    " tokens/s": "tokens/s",
    " img/s": "img/s",
    " images/s": "images/s",
    " req/s": "req/s",
    " samples/s": "samples/s",
}


def _rewrite_rate(call: ast.Call, src: str) -> str | None:
    """fmt/fmt_int(..., suffix=' tokens/s') -> fmt_rate(...)."""
    seg = ast.get_source_segment(src, call) or ""
    if not call.args or "\n" in seg:
        return None
    fname = call.func.id if isinstance(call.func, ast.Name) else (
        call.func.attr if isinstance(call.func, ast.Attribute) else None
    )
    if fname not in {"fmt", "fmt_int"}:
        return None

    suffix = None
    allowed_kwargs = {"precision", "commas", "suffix", "approx", "lower_bound"}
    kw_src: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg not in allowed_kwargs:
            return None
        if kw.arg == "suffix":
            suffix = _const_str(kw.value)
        elif kw.arg:
            val = ast.get_source_segment(src, kw.value)
            if val is None or "\n" in val:
                return None
            kw_src[kw.arg] = val
    if suffix not in _RATE_SUFFIX_INFO:
        return None
    unit = _RATE_SUFFIX_INFO[suffix]

    arg0 = ast.get_source_segment(src, call.args[0])
    if arg0 is None or "\n" in arg0:
        return None
    if fname == "fmt_int":
        arg0 = arg0 if arg0.strip().startswith("round(") else f"round({arg0})"

    tail = ""
    if fname == "fmt_int":
        tail += ", precision=0"
        if "commas" in kw_src:
            tail += f", commas={kw_src['commas']}"
    else:
        if "precision" in kw_src:
            tail += f", precision={kw_src['precision']}"
        if "commas" in kw_src:
            tail += f", commas={kw_src['commas']}"
    for marker in ("approx", "lower_bound"):
        if marker in kw_src:
            tail += f", {marker}={kw_src[marker]}"
    return f"fmt_rate({arg0}, {unit!r}{tail})"


def scan_rate(path: Path):
    """Return (rate_edits, queue) for exact service-rate suffixes."""
    text = path.read_text(encoding="utf-8", errors="replace")
    edits: list[MultEdit] = []
    queue: list[QueueItem] = []
    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in {"fmt", "fmt_int"}:
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix not in _RATE_SUFFIX_INFO:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            new = _rewrite_rate(call, src)
            if new and var:
                edits.append(MultEdit(file_line, var, seg, new))
            else:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "rate", seg.replace("\n", " ⏎ "),
                    "exact service-rate suffix could not be rewritten "
                    "mechanically; convert by hand to fmt_rate(value, unit)."))
    return edits, queue


def _rewrite_fmt_int_multiplier(call, src: str) -> str | None:
    """fmt_int(arg, …, suffix='×') → fmt_multiple(round(arg), precision=0, commas=…).

    fmt_int is documented as fmt(round(val), precision=0, …) with commas=True,
    so this is provably value-equivalent (only the glyph moves to prose).
    """
    if not call.args:
        return None
    arg0 = ast.get_source_segment(src, call.args[0])
    if arg0 is None or "\n" in arg0:
        return None
    if "prefix=" in (ast.get_source_segment(src, call) or ""):
        return None
    commas = "True"  # fmt_int default
    for kw in call.keywords:
        if kw.arg == "commas":
            v = _const_str(kw.value)
            commas = "False" if (isinstance(kw.value, ast.Constant) and kw.value.value is False) else \
                     ("True" if (isinstance(kw.value, ast.Constant) and kw.value.value is True) else commas)
    inner = arg0 if arg0.strip().startswith("round(") else f"round({arg0})"
    return f"fmt_multiple({inner}, precision=0, commas={commas})"


def _rewrite_call_to_multiple(seg: str) -> str | None:
    """Textual rewrite of a single-line ``fmt(...×)`` call → ``fmt_multiple(...)``.

    Only plain ``fmt`` is auto-rewritten: ``fmt`` and ``fmt_multiple`` share the
    exact same precision guard, so dropping ``suffix='×'`` and renaming preserves
    the rendered magnitude byte-for-byte. ``fmt_int`` is deliberately NOT handled
    here — it *rounds* its argument (5.032→"5") whereas ``fmt_multiple`` guards
    against silent rounding, so the conversion is not provably value-identical
    and is routed to the human queue instead.
    """
    if "\n" in seg:
        return None
    if not seg.lstrip().startswith("fmt("):
        return None
    if "prefix=" in seg:
        return None  # fmt_multiple has no prefix= — not a clean rewrite
    new = re.sub(r"^fmt\(", "fmt_multiple(", seg, count=1)
    # drop the suffix kwarg (always comma-preceded; positional factor comes first)
    new = re.sub(r",\s*suffix\s*=\s*(?:'[^']*'|\"[^\"]*\")", "", new, count=1)
    # fmt defaults commas=True but fmt_multiple defaults commas=False; if the
    # original relied on the default, pin commas=True so grouping is preserved.
    if "commas=" not in new and new.endswith(")"):
        new = new[:-1] + ", commas=True)"
    return new


def scan_file(path: Path, variants: bool = False):
    """Return (mult_edits, mult_vars, queue_items) for one chapter.

    variants=False (default): only the provable, value-identical lane —
      fmt(…suffix='×') -> fmt_multiple (exact glyph). Everything else queued.
    variants=True: retired with the public ``multiple`` command. Under Design B,
      multiplier variants should become ``fmt_multiple(...)`` exports named
      ``*_mult_str``; prose uses the computed ref by itself.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    mult_edits: list[MultEdit] = []
    mult_vars: set[str] = set()
    queue: list[QueueItem] = []

    for fence_line, src in extract_python_cells(text):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # map each line to its innermost enclosing class so exports are tracked
        # as Class.attr (prose refs them that way) — avoids the name-collision
        # hazard where two cells export the same bare attr but only one is rewritten.
        classes = [(n.name, n.lineno, n.end_lineno or n.lineno)
                   for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        def _qualifier(lineno: int) -> str:
            best = None
            for name, s, e in classes:
                if s <= lineno <= e and (best is None or s > best[1]):
                    best = (name, s, e)
            return best[0] + "." if best else ""

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            fn = call.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if fname not in REWRITABLE_FNS:
                continue
            suffix = None
            for kw in call.keywords:
                if kw.arg == "suffix":
                    suffix = _const_str(kw.value)
            if suffix is None:
                continue
            var = _assign_target(node)
            seg = ast.get_source_segment(src, call) or ""
            file_line = fence_line + call.lineno
            sfx = suffix.strip()

            if suffix == "×":
                # exact glyph — provable lane for fmt; fmt_int handled in variants
                new = _rewrite_call_to_multiple(seg)
                if not new and variants and var and fname == "fmt_int":
                    new = _rewrite_fmt_int_multiplier(call, src)
                if new and var:
                    mult_edits.append(MultEdit(file_line, var, seg, new))
                    mult_vars.add(_qualifier(node.lineno) + var)
                elif fname == "fmt_int":
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "fmt_int-multiplier", seg.replace("\n", " ⏎ "),
                        "fmt_int rounds; pick fmt_multiple(round(x), precision=0) if "
                        "integer display is intended, else fmt_multiple(x, precision=N). "
                        "Rename the export to *_mult_str and use the computed ref by itself."))
                else:
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "multiline-multiplier", seg.replace("\n", " ⏎ "),
                        "Convert by hand to fmt_multiple(...), rename to *_mult_str, "
                        "and use the computed ref by itself in prose."))
            elif sfx in {"×", "x", "X"}:
                # space-padded glyph (' ×') or a LITERAL letter x, or an fmt_int
                # multiplier. In variants mode these are converted (intended
                # normalization, transformation-gated); otherwise queued.
                new = None
                if variants and var:
                    if fname == "fmt":
                        new = _rewrite_call_to_multiple(seg)
                    elif fname == "fmt_int":
                        new = _rewrite_fmt_int_multiplier(call, src)
                if new and var:
                    mult_edits.append(MultEdit(file_line, var, seg, new))
                    mult_vars.add(_qualifier(node.lineno) + var)
                else:
                    queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                        "multiplier-variant", seg.replace("\n", " ⏎ "),
                        f"suffix={suffix!r}: confirm it is a multiplier, then "
                        "convert to fmt_multiple(...), rename to *_mult_str, and "
                        "drop the literal x/space from prose."))
            elif sfx in {g.strip() for g in PERCENT_GLYPHS}:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "percent", seg.replace("\n", " ⏎ "),
                    "Decide source scale (ratio vs 0-100) → fmt_percent(ratio, "
                    "style=symbol|prose by context). Body=prose, table/eq=symbol."))
            elif sfx in SCALE_GLYPHS:
                queue.append(QueueItem(str(path), file_line, var, fname, suffix,
                    "scale", seg.replace("\n", " ⏎ "),
                    "Move scale division into fmt_count(raw, scale=…) or keep as unit."))

    return mult_edits, mult_vars, queue


def _ref_is_mult(ref: str, mult_vars: set[str]) -> bool:
    """True if a prose ref names a rewritten multiplier export.

    mult_vars holds qualified names ('Class.attr' for class exports, bare
    'attr' for module-level). A prose ref 'Class.attr' must match the qualified
    entry exactly; a bare ref matches a bare entry. This prevents patching a
    same-named export from a *different* class that was not rewritten.
    """
    if ref in mult_vars:
        return True
    # module-level export referenced bare
    return "." not in ref and ref in mult_vars


def _patch_prose(text: str, mult_vars: set[str]) -> tuple[str, list[tuple[int, str]]]:
    """Append $\\times$ after refs to multiplier vars that lack it. Returns
    (new_text, [(line, ref)])."""
    if not mult_vars:
        return text, []
    out_lines: list[str] = []
    patches: list[tuple[int, str]] = []
    in_cell = False
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"^```\{python\}", line):
            in_cell = True
            out_lines.append(line)
            continue
        if in_cell and re.match(r"^```\s*$", line):
            in_cell = False
            out_lines.append(line)
            continue
        if in_cell:
            out_lines.append(line)
            continue

        # collect insertion points (end offsets) for qualifying refs, then splice
        # right-to-left so repeated identical refs on one line each get exactly
        # one $\times$ (str.replace would collide on identical tokens).
        inserts: list[tuple[int, str]] = []
        for m in INLINE_PY.finditer(line):
            ref = m.group(1)
            if not _ref_is_mult(ref, mult_vars):
                continue
            if _TIMES_AFTER.match(line[m.end():]):
                continue  # already followed by the glyph
            inserts.append((m.end(), ref))
        new_line = line
        for pos, ref in sorted(inserts, key=lambda t: -t[0]):
            new_line = new_line[:pos] + "$\\times$" + new_line[pos:]
            patches.append((i, ref))
        out_lines.append(new_line)
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_nl, patches


def _apply_cell_edits(text: str, edits: list[MultEdit]) -> str:
    lines = text.splitlines()
    for e in edits:
        idx = e.line - 1
        if 0 <= idx < len(lines) and e.old in lines[idx]:
            lines[idx] = lines[idx].replace(e.old, e.new, 1)
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing_nl


def cmd_multiple(args) -> int:
    print(
        "codemod_fmt.py multiple is retired. Design B makes fmt_multiple own "
        "$\\times$ and uses *_mult_str exports; use "
        "codemod_design_b_multipliers.py for current migrations.",
        file=sys.stderr,
    )
    return 2

    files = [Path(p) for p in args.qmd]
    total_cell = total_prose = 0
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        edits, mult_vars, _ = scan_file(path)
        new_text = _apply_cell_edits(text, edits)
        new_text, prose_patches = _patch_prose(new_text, mult_vars)
        rel = str(path).split("contents/")[-1]
        if not edits and not prose_patches:
            continue
        print(f"\n=== {rel} ===")
        for e in edits:
            print(f"  cell  L{e.line}  {e.var}")
            print(f"    -  {e.old}")
            print(f"    +  {e.new}")
        for ln, ref in prose_patches:
            print(f"  prose L{ln}  +$\\times$ after `{{python}} {ref}`")
        total_cell += len(edits)
        total_prose += len(prose_patches)
        if args.write:
            path.write_text(new_text, encoding="utf-8")
    verb = "APPLIED" if args.write else "DRY-RUN (use --write to apply)"
    print(f"\n{verb}: {total_cell} cell rewrites, {total_prose} prose $\\times$ insertions.")
    if not args.write and (total_cell or total_prose):
        print("After --write, verify: assess_equiv (values change, visible prose IDENTICAL) "
              "+ fmt_prose_contract (clean).")
    return 0


def cmd_queue(args) -> int:
    files = [Path(p) for p in args.qmd]
    if args.root:
        files += sorted(Path(args.root).rglob("*.qmd"))
    items: list[QueueItem] = []
    for path in files:
        if path.is_file():
            _, _, q = scan_file(path)
            items.extend(q)
    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    payload = {"summary": by_kind, "count": len(items), "items": [asdict(i) for i in items]}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[queue] {len(items)} item(s) -> {args.out}")
    print(f"by kind: {by_kind}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("multiple", help="retired pre-Design-B multiplier relocation")
    m.add_argument("qmd", nargs="+")
    m.add_argument("--write", action="store_true", help="apply edits (default: dry-run)")
    m.set_defaults(func=cmd_multiple)

    q = sub.add_parser("queue", help="emit adjudication queue (percent/scale/multiline)")
    q.add_argument("qmd", nargs="*")
    q.add_argument("--root")
    q.add_argument("--out")
    q.set_defaults(func=cmd_queue)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
