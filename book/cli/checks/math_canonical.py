#!/usr/bin/env python3
"""
Math canonical check — fmt-family and suffix discipline for LEGO cells.

Enforces the canonical math-rendering convention (see the project math rules):

  1. Every ``*_str`` / ``*_math`` / ``*_eq`` / ``*_frac`` assignment in a
     ``{python}`` cell must use ``mlsysim.fmt`` helpers or ``MarkdownStr``.
  2. Every inline ``{python} X`` reference must use a name ending in one of
     those suffixes.
  3. ``fmt()``/``fmt_int()`` outputs must use helper-level
     ``prefix=``/``suffix=`` instead of manual string concatenation.

Invoked by::

    ./book/binder check math --scope canonical

The standalone CLI ``book/tools/audit/audit_math_canonical.py`` is a thin
wrapper around this module for ad-hoc runs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_SUFFIXES = ("_str", "_math", "_eq", "_frac")

# Pattern: inline {python} ref in prose
INLINE_REF = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")

# Pattern: variable assignment to f-string, e.g.  `xxx_str = f"..."`
# The capture group is the variable name (no class-attribute access here —
# assignments inside cells are bare names).
FSTRING_ASSIGN = re.compile(
    r"^(\s*)([A-Za-z_]\w*)\s*=\s*f[\"']"
)

# Pattern: assignment of any kind (used as a sanity catch)
PLAIN_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([^=].*)$")

# Pattern: matches calls to canonical formatter helpers on the RHS.
CANONICAL_STR_CALL = re.compile(
    r"\b(fmt|fmt_qty|fmt_qty_int|fmt_int|fmt_usd|fmt_eur|fmt_percent|fmt_pp|fmt_multiple|fmt_count"
    r"|fmt_ratio|fmt_range|fmt_qty_range|fmt_time_range|fmt_count_range"
    r"|fmt_usd_range|fmt_time|fmt_rate|fmt_fps|fmt_val|fmt_unit|fmt_sci|fmt_sci_qty"
    r"|fmt_magnitude|fmt_text|fmt_display_math"
    r"|fmt_percent_range|fmt_multiple_range"
    r"|fmt_power|fmt_energy|fmt_emissions|fmt_bandwidth|fmt_memory|fmt_latency|fmt_area|fmt_heat_flux|fmt_specific_heat"
    r"|fmt_memory_capacity"
    r"|fmt_params|fmt_tokens|fmt_flop_rate|fmt_flops|fmt_ops_rate"
    r"|fmt_sci_flops|fmt_energy_per_flop|fmt_energy_per_op|fmt_energy_per_byte|fmt_energy_per_bit"
    r"|fmt_decibel|fmt_illuminance|fmt_temperature|fmt_temperature_rate"
    r"|fmt_arithmetic_intensity|fmt_compute_efficiency|fmt_length"
    r"|fmt_carbon_intensity|fmt_water|fmt_water_rate|fmt_water_intensity"
    r"|MarkdownStr)\s*\("
)
CANONICAL_MATH_CALL = re.compile(
    r"\b(fmt_math|fmt_display_math|MarkdownStr)\s*\("
)
CANONICAL_FRAC_CALL = re.compile(r"\b(fmt_frac|MarkdownStr)\s*\(")

# Quarto cell delimiters
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")

# Pattern: a numeric formatter whose first positional arg is itself an
# already-formatted value. The numeric formatters (fmt, fmt_percent, fmt_sci,
# fmt_val, fmt_unit) expect a Number/Quantity; passing a MarkdownStr or str
# from another helper fails at render time with ValueError.
#
# NOTE: fmt_math() and fmt_frac() are EXCLUDED from the outer set because
# they legitimately accept LaTeX strings (e.g., the canonical pattern
# `fmt_math(sci_latex(x))` embeds Unicode scientific notation in inline math).
#
# Caught shapes (all numeric outer):
#   fmt(fmt(x, precision=0), precision=0, prefix="$")           # MarkdownStr
#   fmt(sci_latex(P, precision=1), precision=0, prefix="$")     # plain str
#   fmt(MarkdownStr(label), precision=0)                        # MarkdownStr
#   fmt(gpt3_params_b_str, precision=0, suffix=" billion")      # MarkdownStr
DOUBLE_WRAP_CALL = re.compile(
    r"\b(?:fmt|fmt_usd|fmt_percent|fmt_sci|fmt_val|fmt_unit)\(\s*(?:"
    r"fmt[a-z_]*\(|"          # fmt(fmt_*(...))
    r"sci_latex\(|"           # fmt(sci_latex(...))
    r"MarkdownStr\(|"         # fmt(MarkdownStr(...))
    r"[A-Za-z_]\w*_(?:str|math|eq|frac)\b"  # fmt(VAR_str/math/eq/frac, ...)
    r")"
)

# Pattern: any non-integer literal float assignment, e.g. `devops_fte = 0.1`,
# `edge_inf_cost = 0.001`, `weight_int4_shard_gb = 8.75`. Captures
# (variable, literal value text). The audit function parses the literal
# and flags when the value has a non-zero fractional part AND is later
# formatted with precision=0 — both sub-classes (rounds-to-zero → crash
# at runtime; rounds-to-nonzero-integer → silently wrong arithmetic in
# prose) are real bugs.
NONINT_FLOAT_ASSIGN = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+\.\d+(?:e-?\d+)?)\b"
)

# Pattern: fmt() call with precision=0 (only fmt itself; fmt_int and fmt_percent
# have their own semantics). Captures the variable being formatted.
PRECISION_ZERO_CALL = re.compile(
    r"\bfmt\(\s*([A-Za-z_][\w.]*)\s*,[^)]*\bprecision\s*=\s*0\b"
)

# Pattern: fmt(..., precision=N) with N>=1 on an integer literal first arg.
SPURIOUS_DECIMAL_FMT = re.compile(
    r"\bfmt\(\s*(\d+)\s*,[^)]*\bprecision\s*=\s*([1-9]\d*)\b"
)

# Pattern: fmt(round/int(...), precision=0) — prefer fmt_int(...) at call site.
IMPLICIT_INT_CAST_FMT = re.compile(
    r"\bfmt\(\s*(?:round|int)\("
)

# Pattern: fmt-family helper names used as calls in a cell body. Any of these
# requires a matching `from mlsysim.fmt import ...` line in the file.
FMT_FAMILY_USE = re.compile(
    r"\b(fmt|fmt_qty|fmt_qty_int|fmt_int|fmt_usd|fmt_eur|fmt_math|fmt_percent|fmt_pp|fmt_multiple"
    r"|fmt_count|fmt_ratio|fmt_range|fmt_qty_range|fmt_time_range"
    r"|fmt_count_range|fmt_usd_range|fmt_time|fmt_rate|fmt_fps|fmt_val|fmt_unit"
    r"|fmt_magnitude|fmt_text|fmt_display_math"
    r"|fmt_power|fmt_energy|fmt_emissions|fmt_bandwidth|fmt_memory|fmt_latency|fmt_area|fmt_heat_flux|fmt_specific_heat"
    r"|fmt_memory_capacity"
    r"|fmt_params|fmt_tokens|fmt_flop_rate|fmt_flops|fmt_ops_rate"
    r"|fmt_sci_flops|fmt_energy_per_flop|fmt_energy_per_op|fmt_energy_per_byte|fmt_energy_per_bit"
    r"|fmt_decibel|fmt_illuminance|fmt_temperature|fmt_temperature_rate"
    r"|fmt_arithmetic_intensity|fmt_compute_efficiency|fmt_length"
    r"|fmt_carbon_intensity|fmt_water|fmt_water_rate|fmt_water_intensity"
    r"|fmt_sci|fmt_frac|sci_latex|MarkdownStr|check)\s*\("
)

# Pattern: `from mlsysim.fmt import ...` block (possibly multi-line in parens
# — we collapse parens before matching, so this single-line form is enough).
FMT_IMPORT_LINE = re.compile(
    r"\bfrom\s+mlsysim\.fmt\s+import\s+([^#\n]+)"
)

MLSYSIM_STAR_IMPORT = re.compile(r"\bfrom\s+mlsysim\s+import\s+\*")

# Names exported by `from mlsysim import *` that belong to the fmt family.
MLSYSIM_STAR_FMT_NAMES = frozenset({
    "fmt", "fmt_qty", "fmt_qty_int", "fmt_int", "fmt_usd", "fmt_eur", "fmt_percent", "fmt_pp", "fmt_multiple",
    "fmt_count", "fmt_ratio", "fmt_range", "fmt_qty_range", "fmt_time_range",
    "fmt_count_range", "fmt_usd_range", "fmt_time", "fmt_rate", "fmt_fps", "fmt_val",
    "fmt_unit", "fmt_sci", "fmt_sci_qty", "fmt_magnitude", "fmt_text",
    "fmt_frac", "fmt_math", "fmt_display_math", "MarkdownStr", "check",
    "fmt_percent_range", "fmt_multiple_range",
    "sci_latex",
    "fmt_power", "fmt_energy", "fmt_emissions", "fmt_bandwidth", "fmt_memory",
    "fmt_latency", "fmt_area", "fmt_heat_flux", "fmt_specific_heat", "fmt_memory_capacity",
    "fmt_params", "fmt_tokens", "fmt_flop_rate", "fmt_compute_efficiency",
    "fmt_flops", "fmt_ops_rate", "fmt_arithmetic_intensity", "fmt_length",
    "fmt_sci_flops", "fmt_energy_per_flop", "fmt_energy_per_op", "fmt_energy_per_byte",
    "fmt_energy_per_bit",
    "fmt_decibel", "fmt_illuminance", "fmt_temperature",
    "fmt_temperature_rate",
    "fmt_carbon_intensity", "fmt_water", "fmt_water_rate",
    "fmt_water_intensity",
})

# Pattern: fmt_percent(..., suffix=...) — fmt_percent does not accept suffix=.
# This would raise a TypeError at render time.
FMT_PERCENT_SUFFIX = re.compile(
    r"\bfmt_percent\([^)]*\bsuffix\s*="
)

# Pattern: manual string assembly around formatter helpers that already support
# prefix=/suffix=. These are line-local on purpose; the `str(fmt(...))` branch
# catches the multi-line bug class where a separate line does
# `"\\$" + str(fmt(...))`.
DECORATABLE_FMT_CALL = r"(?:fmt|fmt_int)"
MANUAL_FMT_STR_WRAP = re.compile(
    rf"\bstr\s*\(\s*{DECORATABLE_FMT_CALL}\s*\("
)
MANUAL_FMT_SUFFIX_CONCAT = re.compile(
    rf"\b{DECORATABLE_FMT_CALL}\s*\([^#\n]*\)\s*\+\s*(?:[rRuUbBfF]*)?[\"']"
)
MANUAL_FMT_PREFIX_CONCAT = re.compile(
    rf"(?:[rRuUbBfF]*)?[\"'][^#\n]*[\"']\s*\+\s*(?:str\s*\(\s*)?{DECORATABLE_FMT_CALL}\s*\("
)


@dataclass
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str


def _suffix_of(ref: str) -> str:
    """Return the suffix of the last component of a dotted ref."""
    last = ref.split(".")[-1]
    for s in ALLOWED_SUFFIXES:
        if last.endswith(s):
            return s
    return ""


def _audit_python_cells(qmd_path: Path) -> list[Violation]:
    """Find _str/_math/_eq/_frac assignments inside Python cells that don't
    use the canonical formatter family."""
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue

        # We only check assignments where the variable name has an approved suffix.
        m = PLAIN_ASSIGN.match(line)
        if not m:
            continue
        varname, rhs = m.group(1), m.group(2)
        suffix = _suffix_of(varname)
        if not suffix:
            continue  # name doesn't carry our convention; not our concern

        # Skip "type-hint" reassignment patterns or augmented assignments.
        if "==" in line[:line.index("=") + 2]:
            continue

        # Skip when RHS itself is a function call to a canonical helper.
        if suffix == "_str":
            if CANONICAL_STR_CALL.search(rhs) or CANONICAL_MATH_CALL.search(rhs):
                continue
        elif suffix in ("_math", "_eq"):
            if CANONICAL_MATH_CALL.search(rhs):
                continue
        elif suffix == "_frac":
            if CANONICAL_FRAC_CALL.search(rhs):
                continue

        # Skip when RHS is just another variable (e.g., `x_str = y_str`).
        # That's a re-binding, not a fresh format; the original assignment
        # is what we want to catch.
        if re.match(r"^\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*$", rhs):
            continue

        # Flag: a bare f-string is the most common violation, but ANY
        # non-canonical RHS for a _str-family variable is a violation.
        is_fstring = bool(re.match(r"^\s*f[\"']", rhs))
        code = "noncanonical_fstring_assign" if is_fstring else "noncanonical_str_assign"
        out.append(
            Violation(
                file=rel,
                line=i,
                code=code,
                message=(
                    f"`{varname}` (suffix `{suffix}`) is not built via the "
                    f"canonical helper family (fmt/fmt_int/fmt_usd/fmt_percent/"
                    f"fmt_pp/fmt_multiple/fmt_count/fmt_ratio/fmt_range/fmt_qty/fmt_qty_int/"
                    f"fmt_*_range/fmt_time/fmt_rate/domain fmt helpers/fmt_math/fmt_frac/"
                    f"MarkdownStr). "
                    f"Use the appropriate helper from mlsysim.fmt so the value "
                    f"renders correctly inside math mode."
                ),
                context=line.strip()[:160],
            )
        )

    return out


def _audit_inline_refs(qmd_path: Path) -> list[Violation]:
    """Find `{python} X` references in prose where X (or its last
    component) doesn't end in an approved suffix."""
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
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

        for m in INLINE_REF.finditer(line):
            ref = m.group(1)
            if not _suffix_of(ref):
                out.append(
                    Violation(
                        file=rel,
                        line=i,
                        code="inline_ref_no_canonical_suffix",
                        message=(
                            f"`{{python}} {ref}` does not end in an approved "
                            f"suffix ({', '.join(ALLOWED_SUFFIXES)}). Rename the "
                            f"underlying attribute so the type-of-value is "
                            f"declared by the suffix."
                        ),
                        context=line.strip()[:160],
                    )
                )

    return out


def _audit_double_wrap(qmd_path: Path) -> list[Violation]:
    """Find fmt-family calls whose first positional arg is itself a fmt-family
    call or a _str/_math/_eq/_frac variable.

    Both shapes pass a MarkdownStr into fmt(), which fails at render time
    with ``ValueError: Unknown format code 'f' for object of type 'MarkdownStr'``.
    Seen in 2026-05 codemod artifacts.
    """
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue
        m = DOUBLE_WRAP_CALL.search(line)
        if not m:
            continue
        out.append(
            Violation(
                file=rel,
                line=i,
                code="double_wrap_fmt",
                message=(
                    "Double-wrap detected: a fmt-family call receives a "
                    "MarkdownStr (either another fmt(...) call or a "
                    "_str/_math/_eq/_frac variable) as its first arg. The "
                    "outer fmt() will raise ValueError at render time. "
                    "Pass the underlying numeric value directly, and use "
                    "prefix=/suffix= to compose the final string."
                ),
                context=line.strip()[:160],
            )
        )
    return out


def _audit_precision_loss_on_small_floats(qmd_path: Path) -> list[Violation]:
    """Find fmt(VAR, precision=0, ...) calls where VAR is assigned a non-
    integer literal float — i.e., a value with a meaningful fractional
    part where precision=0 silently discards the author's intent.

    Two sub-classes:
    (a) value rounds to '0' at precision=0 → trips fmt()'s runtime guard
        (loud failure — ValueError at render time). Examples: 0.001, 0.4.
    (b) value rounds to a non-zero integer → guard does NOT fire (silent
        failure — the rendered prose carries the wrong number, which is
        worse). Examples:
          • `util_peak = 0.85` → "1" → prose `1 × 312 = 265` is false
          • `weight_int4_shard_gb = 8.75` → "9" → prose `80 - 9 = 71.25`
            visibly contradicts itself.

    Both sub-classes are real bugs. The author wrote a non-integer literal
    intending to display its fractional value; rounding it to integer
    discards that intent regardless of which integer it lands on.
    """
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    cell_assignments: dict[str, tuple[int, float]] = {}
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            cell_assignments = {}
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue

        # Record any non-integer literal-float assignment. Skip values that
        # parse to an integer (e.g., 100.0) — formatting those at
        # precision=0 produces the right display and is not a bug.
        m_assign = NONINT_FLOAT_ASSIGN.match(line)
        if m_assign:
            varname = m_assign.group(1)
            try:
                value = float(m_assign.group(2))
            except ValueError:
                continue
            if value != int(value):
                cell_assignments[varname] = (i, value)
            continue

        # Check fmt(VAR, precision=0, ...) calls.
        m_call = PRECISION_ZERO_CALL.search(line)
        if not m_call:
            continue
        var = m_call.group(1).split(".")[-1]
        if var not in cell_assignments:
            continue
        assign_line, value = cell_assignments[var]
        rounded = f"{value:.0f}"
        if rounded == "0":
            kind = (
                f"Rounding `{value}` to integer produces '0', which trips "
                f"fmt()'s precision-loss guard at render time (loud "
                f"failure — ValueError)."
            )
        else:
            kind = (
                f"Rounding `{value}` to integer produces '{rounded}', "
                f"which does NOT trip fmt()'s guard but silently displays "
                f"the wrong value. If the prose uses this in arithmetic "
                f"(e.g., `{var} × X = Y`), the rendered equation becomes "
                f"nonsense (`{rounded} × X = Y` is false unless Y matches)."
            )
        out.append(
            Violation(
                file=rel,
                line=i,
                code="precision_zero_on_small_float",
                message=(
                    f"`{var}` was assigned `{value}` at line {assign_line} "
                    f"and is formatted with `precision=0`. {kind} Fix "
                    f"options: (a) bump precision to preserve the value "
                    f"(e.g., `precision=2` for `{value}`), (b) use "
                    f"`fmt_int({var})` if integer rounding to `{rounded}` "
                    f"is genuinely intentional, or (c) use "
                    f"`MarkdownStr(f\"{{x}}\")` for value ranges that "
                    f"span orders of magnitude."
                ),
                context=line.strip()[:160],
            )
        )
    return out


def _audit_spurious_decimal_precision(qmd_path: Path) -> list[Violation]:
    """Find fmt(INTEGER_LITERAL, precision>=1) — produces spurious .0 at runtime."""
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue

        m = SPURIOUS_DECIMAL_FMT.search(line)
        if not m:
            continue
        literal, prec = m.group(1), m.group(2)
        out.append(
            Violation(
                file=rel,
                line=i,
                code="spurious_decimal_precision",
                message=(
                    f"Integer literal `{literal}` is formatted with "
                    f"`precision={prec}`, which produces spurious trailing "
                    f"zeros (e.g. `{literal}.{'0' * int(prec)}`). Use "
                    f"`precision=0` or `fmt_int({literal})` instead."
                ),
                context=line.strip()[:160],
            )
        )
    return out


def _audit_implicit_int_cast_precision_zero(qmd_path: Path) -> list[Violation]:
    """Prefer fmt_int(...) over fmt(round/int(...), precision=0) for explicit intent."""
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue
        if not IMPLICIT_INT_CAST_FMT.search(line):
            continue
        if "precision=0" not in line:
            continue
        cast = "int(...)" if re.search(r"\bfmt\(\s*int\(", line) else "round(...)"
        out.append(
            Violation(
                file=rel,
                line=i,
                code="prefer_fmt_int",
                message=(
                    f"Use `fmt_int(...)` instead of `fmt({cast}, precision=0)` "
                    f"so integer display intent is explicit at the call site."
                ),
                context=line.strip()[:160],
            )
        )
    return out


def _audit_missing_fmt_imports(qmd_path: Path) -> list[Violation]:
    """Find files where a fmt-family helper is called in a cell before any
    cell imports it from mlsysim.fmt.

    Quarto cells share a Jupyter kernel namespace, so an import in an
    earlier cell satisfies later uses — but a use in cell N that imports
    only in cell N+1 fails at execution time. This check walks cells in
    document order, accumulating imported names, and flags any helper
    call whose name is not yet in the cumulative import set.
    """
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    imported: set[str] = set()  # cumulative across cells in document order
    flagged: set[tuple[str, int]] = set()  # dedupe per (name, line)

    # Buffer to handle multi-line `from mlsysim.fmt import ( ... )`. When we
    # see the opening line, accumulate until the closing `)`.
    pending_import_buffer: list[str] = []
    in_paren_import = False

    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            in_paren_import = False
            pending_import_buffer = []
            continue
        if not in_cell:
            continue

        # Multi-line paren import handling.
        if in_paren_import:
            pending_import_buffer.append(line)
            if ")" in line:
                blob = " ".join(pending_import_buffer)
                for name in re.findall(r"[A-Za-z_]\w*", blob):
                    if name not in ("as", "import", "from", "mlsysim", "fmt"):
                        imported.add(name)
                in_paren_import = False
                pending_import_buffer = []
            continue

        # Single-line `from mlsysim.fmt import ...` (with or without parens).
        m_imp = FMT_IMPORT_LINE.search(line)
        if m_imp:
            blob = m_imp.group(1)
            if "(" in blob and ")" not in blob:
                pending_import_buffer = [blob]
                in_paren_import = True
                continue
            for name in re.findall(r"[A-Za-z_]\w*", blob):
                if name != "as":
                    imported.add(name)
            continue

        if MLSYSIM_STAR_IMPORT.search(line):
            imported.update(MLSYSIM_STAR_FMT_NAMES)
            continue

        # Skip comment lines.
        if line.strip().startswith("#"):
            continue

        # Check fmt-family uses.
        for m in FMT_FAMILY_USE.finditer(line):
            name = m.group(1)
            if name in imported:
                continue
            if (name, i) in flagged:
                continue
            flagged.add((name, i))
            out.append(
                Violation(
                    file=rel,
                    line=i,
                    code="missing_fmt_import",
                    message=(
                        f"`{name}(...)` is used here but `{name}` has not yet "
                        f"been imported from `mlsysim.fmt` in any prior cell. "
                        f"Quarto evaluates cells in document order; add "
                        f"`{name}` to this cell's import block (or an earlier "
                        f"cell's) so the call resolves at exec time."
                    ),
                    context=line.strip()[:160],
                )
            )
    return out


def _audit_fmt_percent_suffix(qmd_path: Path) -> list[Violation]:
    """Flag ``fmt_percent(..., suffix=...)`` calls.

    ``fmt_percent()`` does not accept a ``suffix=`` keyword argument.  The
    percent glyph is owned by the formatter via ``style=`` (``'prose'`` →
    "85 percent", ``'symbol'`` → "85%", ``'number'`` → "85").  Passing
    ``suffix=`` raises ``TypeError`` at render time.
    """
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue
        if FMT_PERCENT_SUFFIX.search(line):
            out.append(
                Violation(
                    file=rel,
                    line=i,
                    code="fmt_percent_suffix",
                    message=(
                        "fmt_percent() does not accept suffix=. The percent "
                        "glyph is owned by the formatter: pass style='prose' "
                        "(\"85 percent\") or style='symbol' (\"85%\") instead "
                        "of a suffix."
                    ),
                    context=line.strip()[:160],
                )
            )
    return out


def _audit_manual_fmt_string_assembly(qmd_path: Path) -> list[Violation]:
    """Flag manual prefix/suffix assembly around fmt-family outputs.

    `fmt()` and `fmt_int()` return MarkdownStr and already support
    `prefix=`/`suffix=`. Manually wrapping them with `str(...)` or
    concatenating literal strings recreates the old escape-hatch style and is
    easy to break in Quarto math/prose contexts.
    """
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    in_cell = False
    for i, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if not in_cell:
            continue
        if line.strip().startswith("#"):
            continue

        if MANUAL_FMT_STR_WRAP.search(line):
            reason = "str(fmt(...))"
        elif MANUAL_FMT_SUFFIX_CONCAT.search(line):
            reason = "fmt(...) + literal"
        elif MANUAL_FMT_PREFIX_CONCAT.search(line):
            reason = "literal + fmt(...)"
        else:
            continue

        out.append(
            Violation(
                file=rel,
                line=i,
                code="manual_fmt_string_assembly",
                message=(
                    f"Manual formatter string assembly detected ({reason}). "
                    "Use `prefix=` and/or `suffix=` on fmt()/fmt_int() "
                    "instead of concatenating or coercing the formatter output."
                ),
                context=line.strip()[:160],
            )
        )
    return out


def _extract_python_cells(lines: list[str]):
    """Yield ``(fence_line, source)`` for each Python cell in a QMD file."""
    in_cell = False
    start = 0
    body: list[str] = []
    for i, line in enumerate(lines, 1):
        if not in_cell and CELL_START.match(line):
            in_cell = True
            start = i
            body = []
            continue
        if in_cell and CELL_END.match(line):
            yield start, "\n".join(body)
            in_cell = False
            body = []
            continue
        if in_cell:
            body.append(line)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _string_template(node: ast.AST) -> str:
    """Return literal text with formatted slots replaced by ``{}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _string_template(node.left) + _string_template(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        # Preserve the common MarkdownStr((rf"$...$").replace(...)) shape.
        return _string_template(node.func.value)
    return ""


def _audit_manual_markdown_math(qmd_path: Path) -> list[Violation]:
    """Flag MarkdownStr("$...$") and MarkdownStr("$$...$$") math wrappers."""
    out: list[Violation] = []
    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    rel = str(qmd_path)
    for fence_line, src in _extract_python_cells(lines):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node.func) != "MarkdownStr" or not node.args:
                continue
            template = _string_template(node.args[0]).strip()
            if not (template.startswith("$") and template.endswith("$")):
                continue
            helper = "fmt_display_math" if template.startswith("$$") else "fmt_math"
            out.append(
                Violation(
                    file=rel,
                    line=fence_line + node.lineno,
                    code="manual_markdownstr_math",
                    message=(
                        f"Manual MarkdownStr math wrapper detected. Use `{helper}(...)` "
                        "so math rendering goes through the canonical formatter API."
                    ),
                    context=(ast.get_source_segment(src, node) or "").replace("\n", " ")[:160],
                )
            )
    return out


def audit(paths: list[Path]) -> list[Violation]:
    all_violations: list[Violation] = []
    for p in paths:
        if p.is_file():
            files = [p]
        else:
            files = sorted(p.rglob("*.qmd"))
        for f in files:
            all_violations.extend(_audit_python_cells(f))
            all_violations.extend(_audit_inline_refs(f))
            all_violations.extend(_audit_double_wrap(f))
            all_violations.extend(_audit_precision_loss_on_small_floats(f))
            all_violations.extend(_audit_spurious_decimal_precision(f))
            all_violations.extend(_audit_implicit_int_cast_precision_zero(f))
            all_violations.extend(_audit_missing_fmt_imports(f))
            all_violations.extend(_audit_fmt_percent_suffix(f))
            all_violations.extend(_audit_manual_fmt_string_assembly(f))
            all_violations.extend(_audit_manual_markdown_math(f))
    return all_violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="QMD files or directories. Defaults to book/quarto/contents/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human report.",
    )
    parser.add_argument(
        "--by-file",
        action="store_true",
        help="Aggregate counts by file; suppress per-violation detail.",
    )
    args = parser.parse_args()

    targets = args.paths or [Path("book/quarto/contents")]
    violations = audit(targets)

    if args.json:
        out = [
            {
                "file": v.file,
                "line": v.line,
                "code": v.code,
                "message": v.message,
                "context": v.context,
            }
            for v in violations
        ]
        print(json.dumps(out, indent=2))
        return 1 if violations else 0

    if args.by_file:
        by_file: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for v in violations:
            by_file[v.file][v.code] += 1
        codes = sorted({v.code for v in violations})
        header = f"{'file':<70s}" + "".join(f"  {c:<28s}" for c in codes) + "  total"
        print(header)
        print("-" * len(header))
        for f in sorted(by_file):
            row = f"{f:<70s}"
            total = 0
            for c in codes:
                n = by_file[f][c]
                total += n
                row += f"  {n:<28d}"
            row += f"  {total}"
            print(row)
        print(f"\nTotal violations: {len(violations)} across {len(by_file)} files")
        return 1 if violations else 0

    # Default: detailed per-violation report.
    for v in violations:
        print(f"{v.file}:{v.line}  [{v.code}]  {v.message}")
        print(f"  {v.context}")
        print()
    print(f"Total violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
