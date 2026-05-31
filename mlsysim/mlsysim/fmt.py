"""
fmt.py
Formatting + presentation helpers for QMD output.
Keep science in mlsysim/physics/; keep display here.
"""

from .core.constants import ureg


class MarkdownStr(str):
    """A string that ALSO renders as raw Markdown when consumed by Quarto/Jupyter.

    Quarto's inline ``{python} x`` substitution escapes commas and decimals in
    plain ``str`` outputs (``2,039.4`` → ``2\\,039\\.4``). Outside math mode the
    escapes are harmless (``\\.`` reads as a literal period), but **inside**
    ``$...$`` math mode they become LaTeX commands — ``\\,`` is ``\\thinspace``
    and ``\\.`` is a dot accent — and the value is silently corrupted to
    ``2 0394``.

    Quarto detects ``_repr_markdown_`` and inserts the content **verbatim**
    without escaping. By subclassing ``str``, every string operation
    (``f"{x}"``, ``x + y``, ``x.replace(...)``, ``len(x)``, slicing) continues
    to work, so this drop-in replacement for plain-``str`` formatters is fully
    backward-compatible with existing call sites.
    """

    def _repr_markdown_(self):
        """Jupyter notebook hook to render the object as Markdown."""
        return str.__str__(self)


def _get_markdown():
    """Return the MarkdownStr class. Retained for backward compatibility."""
    return MarkdownStr


def _require_finite(val, what="value"):
    """Reject non-finite inputs (``inf``/``nan``) with a clear message.

    A divide-by-zero (``x / 0``) yields ``inf``; an indeterminate ``0 / 0``
    yields ``nan``. Without this guard those silently render as the literal
    text ``"inf"``/``"nan"`` (or trip an incidental ``OverflowError`` deep in
    ``round()``). This is the formatter's last-line-of-defense against an
    upstream divide-by-zero leaking into prose. Returns ``val`` unchanged when
    finite so it composes inline.
    """
    import math

    if isinstance(val, float) and not math.isfinite(val):
        kind = "nan (0/0 or undefined)" if math.isnan(val) else "infinite (divide-by-zero)"
        raise ValueError(
            f"Non-finite {what}: {val} is {kind}. The formatter refuses to "
            f"render it — this almost always means a divide-by-zero or an "
            f"undefined 0/0 upstream in the compute. Fix the calculation (or "
            f"guard the denominator) rather than formatting a non-finite value."
        )
    return val


def _numeric_magnitude(quantity):
    """Return a plain float magnitude for fmt safety checks.

    All numeric formatters funnel their value extraction through here, so the
    finite guard applied at this chokepoint protects every helper that takes a
    number or Quantity.
    """
    from .core.provenance import Sourced, scalar_value

    if isinstance(quantity, ureg.Quantity):
        return _require_finite(float(quantity.magnitude), "quantity magnitude")
    if isinstance(quantity, Sourced):
        return _require_finite(float(scalar_value(quantity)), "value")
    return _require_finite(float(quantity), "value")


def _parse_formatted_number(result):
    """Parse a formatted numeric string, ignoring thousands separators."""
    try:
        return float(result.replace(",", ""))
    except ValueError:
        return None


def _is_integer_like(val):
    """True when a float is effectively a whole number (incl. 512.0, 989.0)."""
    return abs(val - round(val)) <= 1e-9


def _has_spurious_zero_decimals(result):
    """True when fixed-point formatting produced trailing .0… with no other digits."""
    plain = result.replace(",", "")
    if "." not in plain:
        return False
    int_part, frac_part = plain.split(".", 1)
    if not int_part or not int_part.lstrip("-").isdigit():
        return False
    return bool(frac_part) and set(frac_part) == {"0"}


def _check_fmt_precision(val, precision, result):
    """Fail loudly when formatting would hide meaningful magnitude.

    Three failure modes:
    1. Any precision: non-zero value displayed as ``0`` (e.g. 0.1 → "0").
    2. precision=0: non-integer value displayed as an integer (e.g. 10.7 → "11").
    3. precision>=1: integer-like value displayed with spurious decimals
       (e.g. 512.0 → "512.0" instead of "512").
    """
    numeric_result = _parse_formatted_number(result)
    if numeric_result is None:
        return

    if numeric_result == 0.0 and abs(val) > 1e-12:
        raise ValueError(
            f"Formatting Precision Error: Value {val} was formatted as '{result}' "
            f"with precision={precision}. This hides the actual value. "
            f"Increase precision or change units to avoid representing a "
            f"non-zero value as zero."
        )

    if precision == 0 and abs(val) > 1e-12:
        nearest_int = round(val)
        if abs(val - nearest_int) > 1e-9:
            raise ValueError(
                f"Formatting Precision Error: Value {val} is not integer-like "
                f"but precision=0 formats it as '{result}'. "
                f"Use precision>=1 to preserve the fractional part, or "
                f"fmt_int({val!r}) if integer display is intentional."
            )

    if precision >= 1 and _is_integer_like(val) and _has_spurious_zero_decimals(result):
        raise ValueError(
            f"Formatting Precision Error: Value {val} is integer-like "
            f"but precision={precision} formats it as '{result}'. "
            f"Use precision=0 or fmt_int(...) to avoid spurious trailing zeros."
        )


def _display_prefix(prefix="", *, approx=False, lower_bound=False):
    """Build a standardized leading marker for a formatted display value."""
    if approx and lower_bound:
        raise ValueError("A value cannot be both approximate and a lower bound.")
    marker = "~" if approx else ("> " if lower_bound else "")
    if prefix and marker:
        raise ValueError(
            "Use either prefix= or the named approx/lower_bound marker, not both."
        )
    return prefix or marker


def fmt(quantity, unit=None, precision=1, commas=True,
        prefix="", suffix="", approx=False, lower_bound=False):
    """
    Format a Pint Quantity (or plain number) for narrative text.
    Returns a MarkdownStr so Quarto inserts the value verbatim (no escape).

    The prefix and suffix arguments collapse the old MarkdownStr(f"...")
    escape-hatch idiom into a single canonical helper. Common uses:

        fmt(price, precision=0, prefix="\\$")      # internal use by fmt_usd
        fmt(value, precision=0, approx=True)       # "~1,000"
        fmt(value, precision=0, lower_bound=True)  # "> 1,000"
        fmt(rate * 100, precision=1, commas=False, suffix="%")  # "12.4%"
        fmt(bw_mb_s, precision=1, commas=False, suffix=" MB/s")  # "2.4 MB/s"
        fmt(speedup, precision=0, commas=False)     # prose adds "$\\times$"

    Safety: Raises ValueError if formatting would hide meaningful magnitude:
    non-zero values displayed as ``0``, non-integers displayed with
    ``precision=0``, or integer-like values shown with spurious decimals
    (``512.0``). Use ``fmt_int(...)`` when integer display is intentional.
    """
    if unit:
        # If a raw number is passed, assume it is already in base units.
        if isinstance(quantity, ureg.Quantity):
            quantity = quantity.to(unit)

    val = _numeric_magnitude(quantity)

    # Primary formatting
    fmt_str = f",.{precision}f" if commas else f".{precision}f"
    result = f"{val:{fmt_str}}"

    _check_fmt_precision(val, precision, result)

    prefix = _display_prefix(prefix, approx=approx, lower_bound=lower_bound)
    decorated = f"{prefix}{result}{suffix}"
    out = MarkdownStr(decorated)
    assert isinstance(out, MarkdownStr), (
        "fmt() must return MarkdownStr — this guard exists so a future refactor "
        "of this module cannot silently break Quarto's _repr_markdown_ detection. "
        "See the project math rules."
    )
    return out


def fmt_int(
    quantity,
    unit=None,
    commas=True,
    prefix="",
    suffix="",
    approx=False,
    lower_bound=False,
):
    """
    Format a value as an integer for narrative text.

    Explicit source-level opt-in for integer display of computed values.
    Equivalent to ``fmt(round(val), precision=0, ...)`` but makes editorial
    intent visible at the call site.
    """
    if unit:
        if isinstance(quantity, ureg.Quantity):
            quantity = quantity.to(unit)
    val = _numeric_magnitude(quantity)
    return fmt(
        round(val),
        precision=0,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


def fmt_usd(amount, *, precision=0, commas=True, approx=False, suffix=""):
    """
    Canonical currency formatter — the single, blessed way to render any
    dollar amount in QMD prose, tables, or callouts.

    This is the currency member of the ``fmt_*`` family (cf. ``fmt_percent``
    for percentages). It exists so the Pandoc/LaTeX escaping detail of a prose
    dollar sign lives in exactly one place: a bare ``$`` in body prose opens a
    math span and silently swallows downstream tokens, so currency must render
    as the escaped ``\\$`` (see ``.claude/rules/numbers-and-math-in-prose.md``
    §4). Authors never type a dollar sign and never type ``prefix=``; they call
    ``fmt_usd(...)`` and the escaping, the optional ``~`` approximation marker,
    and integer rounding are all handled here.

    The literal string ``USD`` is never emitted; the currency code is defined
    once in ``vol1/frontmatter/_notation_body.qmd``.

        fmt_usd(15000)                       # "\\$15,000"
        fmt_usd(c_total, approx=True,
                suffix="/year")              # "~\\$1,234/year"
        fmt_usd(gpt3_cost_m, precision=1,
                suffix="M")                  # "\\$4.6M"
        fmt_usd(rate_per_gb, precision=2,
                commas=False, suffix="/GB")  # "\\$0.09/GB"

    Args:
        amount: A plain number or a pure-dollar Pint ``Quantity`` (converted
            via the ``USD`` unit). For rate values (e.g. dollars per GB), pass
            the already-extracted magnitude and describe the denominator with
            ``suffix`` (e.g. ``suffix="/GB"``).
        precision: Decimal places. ``precision=0`` rounds to whole dollars
            (``fmt_int`` semantics); ``precision>=1`` uses ``fmt`` semantics
            with its spurious-zero guard.
        commas: Thousands separators (default ``True`` — currency usually
            groups: ``\\$15,000``). Pass ``False`` for small rates.
        approx: When ``True``, prepend ``~`` before the dollar sign.
        suffix: Scale glyph (``"K"``/``"M"``/``"B"``) or rate denominator
            (``"/hr"``, ``"/GB/month"``, ``"/year"``). Must not contain a
            currency code; the checker forbids literal ``USD`` here.
    """
    from .core.units import USD

    if isinstance(amount, ureg.Quantity):
        amount = amount.m_as(USD)

    prefix = "~\\$" if approx else "\\$"

    if precision == 0:
        return fmt_int(amount, commas=commas, prefix=prefix, suffix=suffix)
    return fmt(amount, precision=precision, commas=commas, prefix=prefix, suffix=suffix)


def fmt_val(quantity, default="-"):
    """
    Format the magnitude of a Pint Quantity (or a plain scalar) using Python's
    ``:g`` general format — compact, no trailing zeros, variable precision.

    Returns a MarkdownStr. Pairs with ``fmt_unit()`` for side-by-side
    value/unit table columns where ``fmt()``'s fixed-precision output is
    too rigid (e.g., a column that mixes ``2``, ``2.5``, ``80``, ``9.5e10``).

    >>> fmt_val(2.0)        # "2"
    >>> fmt_val(2.5)        # "2.5"
    >>> fmt_val(quantity)   # "80" for 80 GB; "9.5e+10" for 95 GFLOPS as TFLOPS
    """
    if isinstance(quantity, ureg.Quantity):
        val = quantity.magnitude
    else:
        val = quantity
    if val is None:
        return MarkdownStr(default)
    _require_finite(float(val), "table value")
    out = MarkdownStr(f"{val:g}")
    assert isinstance(out, MarkdownStr), "fmt_val() must return MarkdownStr"
    return out


def fmt_unit(quantity, default="-"):
    """
    Extract the unit string of a Pint Quantity. Returns a MarkdownStr.
    For non-Pint values, returns ``default`` wrapped in MarkdownStr.

    Pairs with ``fmt_val()`` for value/unit table columns.

    >>> fmt_unit(80 * GB)        # "gigabyte"
    >>> fmt_unit(80)             # "-"
    """
    if isinstance(quantity, ureg.Quantity):
        unit_label = f"{quantity.units}"
        for plural_rate, canonical_rate in {
            "MFLOPs/s": "MFLOP/s",
            "GFLOPs/s": "GFLOP/s",
            "TFLOPs/s": "TFLOP/s",
            "PFLOPs/s": "PFLOP/s",
            "ZFLOPs/s": "ZFLOP/s",
        }.items():
            unit_label = unit_label.replace(plural_rate, canonical_rate)
        out = MarkdownStr(unit_label)
    else:
        out = MarkdownStr(default)
    assert isinstance(out, MarkdownStr), "fmt_unit() must return MarkdownStr"
    return out


def fmt_percent(ratio, precision=1, commas=False, style="number",
                max_ratio=1.5, allow_negative=False):
    """
    Format a 0-1 **ratio** as a percentage. The single canonical domain for
    percentages: the input is always a fraction in ``[0, 1]`` (``0.85`` →
    ``85``), never an already-scaled ``0-100`` value.

    This one-domain rule is the structural fix for the "no 10,000%" guarantee.
    A value outside ``[0, max_ratio]`` is almost always an already-scaled value
    passed by mistake (``0.85`` accidentally typed/derived as ``85``), so it
    raises instead of silently rendering ``8500``.     Legitimate values above
    ``max_ratio`` (e.g. >150% growth) must pass ``max_ratio=`` explicitly,
    making the intent visible at the call site. Likewise a percentage that can
    be **negative** — a signed rate of change such as ROI or cost delta, as
    opposed to a bounded proportion like accuracy — must pass
    ``allow_negative=True`` (which widens the domain to ``[-max_ratio, max_ratio]``).
    The default refusal of negatives keeps the common proportion case strict.

    The formatter owns the trailing glyph via ``style`` so authors never type
    ``%`` or "percent" in prose (where a stray ``%`` could leak into math mode
    and where MIT Press style — spell out "percent" in body, "%" in tables —
    is otherwise unenforceable):

        style="prose"   →  "85 percent"   (MIT Press body style)
        style="symbol"  →  "85%"          (tables, equations, captions)
        style="number"  →  "85"           (default; prose carries the meaning,
                                            e.g. "85 MFU", "85 goodput")

    Accepts a Pint Quantity (dimensionless magnitude) or plain float.
    """
    if isinstance(ratio, ureg.Quantity):
        r = float(ratio.m_as(''))
    else:
        r = float(ratio)
    _require_finite(r, "percent ratio")

    lo = -max_ratio if allow_negative else 0.0
    if not (lo <= r <= max_ratio):
        raise ValueError(
            f"fmt_percent expects a 0-1 ratio, got {r}. If this is an "
            f"already-scaled 0-100 value, divide by 100 at the source so the "
            f"value flows through fmt_percent as a ratio. If a value above "
            f"{max_ratio * 100:.0f}% is genuinely intended (e.g. growth), pass "
            f"max_ratio= explicitly; if it can be negative (e.g. ROI, a signed "
            f"change), pass allow_negative=True. This guard prevents silent 100x "
            f"errors such as 0.85 -> '8500 percent'."
        )

    if style not in {"number", "prose", "symbol"}:
        raise ValueError(
            f"fmt_percent style must be 'number', 'prose', or 'symbol', "
            f"got {style!r}."
        )
    glyph = {"number": "", "prose": " percent", "symbol": "%"}[style]
    return fmt(r * 100, precision=precision, commas=commas, suffix=glyph)


def fmt_pp(points, precision=1, commas=False, style="prose", attributive=False):
    """
    Format a difference of two percentages as **percentage points**.

    Distinct from ``fmt_percent``: the input is already on the ``0-100`` point
    scale (it is a difference, not a ratio), so it is NOT multiplied by 100.
    Using a dedicated helper keeps "5 percentage points" (an additive gap)
    from being confused with "5 percent" (a multiplicative share).

        style="prose"               →  "7 percentage points"  (plural noun)
        style="prose", value == 1   →  "1 percentage point"    (singular noun)
        style="prose", attributive  →  "7 percentage-point"    (hyphenated
                                        compound adjective, e.g. "a 7
                                        percentage-point gap" — always hyphenated
                                        and singular, per standard English)
        style="symbol"              →  "7 pp"

    The noun form agrees in number with the **rendered** value, so a value that
    rounds to 1 reads "1 percentage point" while 0.9 / 1.5 read "...points".
    Use ``attributive=True`` only when the value directly modifies a following
    noun; use the default noun form everywhere else.
    """
    if isinstance(points, ureg.Quantity):
        v = float(points.m_as(''))
    else:
        v = float(points)
    _require_finite(v, "percentage points")
    if style not in {"prose", "symbol"}:
        raise ValueError(
            f"fmt_pp style must be 'prose' or 'symbol', got {style!r}."
        )
    if attributive and style != "prose":
        raise ValueError(
            "fmt_pp(attributive=True) applies only to style='prose' (the "
            "hyphenated adjective form), not style='symbol'."
        )
    num = str(fmt(v, precision=precision, commas=commas))
    if style == "symbol":
        return MarkdownStr(f"{num} pp")
    if attributive:
        return MarkdownStr(f"{num} percentage-point")
    word = "percentage point" if num == "1" else "percentage points"
    return MarkdownStr(f"{num} {word}")


def fmt_multiple(factor, precision=1, commas=False):
    """
    Format a multiplier / speedup / scaling factor as a **number only**.

    The multiplier glyph belongs in prose as LaTeX ``$\\times$``, never inside
    the computed string (see .claude/rules/math.md §6 #14). This helper exists
    so the value-kind is explicit at the source while keeping the glyph out of
    the string:

        speedup_str = fmt_multiple(3.2)            # "3.2"
        # prose: `{python} speedup_str`$\\times$ faster

    Guard: a multiplier is non-negative.
    """
    v = _numeric_magnitude(factor)
    if v < 0:
        raise ValueError(
            f"fmt_multiple expects a non-negative factor, got {v}. A negative "
            f"multiplier almost always signals a sign or ordering bug."
        )
    return fmt(v, precision=precision, commas=commas)


_COUNT_SCALES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def fmt_count(
    value,
    scale=None,
    precision=0,
    commas=True,
    suffix="",
    approx=False,
    lower_bound=False,
):
    """
    Format a **count** (a dimensionless tally of things), optionally with a
    magnitude scale glyph.

    Replaces the ``fmt(value / MILLION, suffix="M")`` idiom, where the scale
    division and the glyph were two disconnected facts a reader had to keep in
    sync. Here the scale is declared once and applied for you:

        fmt_count(5_300_000, scale="M")              # "5M"
        fmt_count(5_300_000, scale="M", precision=1) # "5.3M"
        fmt_count(70e9, scale="B")                   # "70B"   (e.g. params)
        fmt_count(8192)                              # "8,192" (no scale)
        fmt_count(1024, suffix=" GPUs")              # "1,024 GPUs"
        fmt_count(1024, suffix=" GPUs", approx=True) # "~1,024 GPUs"

    Guard: counts are non-negative.

    For a currency amount use ``fmt_usd``; for a physical quantity use
    ``fmt_qty``. ``fmt_count`` is for pure tallies only.
    """
    v = _numeric_magnitude(value)
    if v < 0:
        raise ValueError(
            f"fmt_count expects a non-negative count, got {v}."
        )
    glyph = ""
    if scale is not None:
        if scale not in _COUNT_SCALES:
            raise ValueError(
                f"fmt_count scale must be one of {sorted(_COUNT_SCALES)} or "
                f"None, got {scale!r}."
            )
        v = v / _COUNT_SCALES[scale]
        glyph = scale
    return fmt(
        v,
        precision=precision,
        commas=commas,
        suffix=glyph + suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


def fmt_ratio(value, precision=1, commas=False, allow_negative=False):
    """
    Format a **dimensionless ratio** as a bare number (no unit, no glyph).

    For quantities that are conceptually a quotient of like things —
    arithmetic intensity (FLOP/byte), compression ratio, overcommit ratio,
    speedup expressed as a plain ratio rather than a "×" multiplier. Naming
    the kind at the call site makes intent explicit and, unlike ``fmt()``,
    ``fmt_ratio`` accepts **no** ``prefix=``/``suffix=`` — a ratio is unitless
    by definition, so decoration is a sign the wrong helper was chosen
    (use ``fmt_qty`` for a dimensioned value, ``fmt_multiple`` for a "×"
    factor, ``fmt_percent`` for a share).

        intensity_str = fmt_ratio(5.0, precision=0)     # "5"    (FLOP/byte)
        compression_str = fmt_ratio(3.2)                # "3.2"

    Guards: finite (inherited) and non-negative by default — a negative ratio
    of magnitudes usually signals a sign bug; pass ``allow_negative=True`` for
    the rare signed ratio.
    """
    v = _numeric_magnitude(value)
    if v < 0 and not allow_negative:
        raise ValueError(
            f"fmt_ratio expects a non-negative ratio, got {v}. Pass "
            f"allow_negative=True if a signed ratio is genuinely intended."
        )
    return fmt(v, precision=precision, commas=commas)


def fmt_range(lo, hi, *, precision=1, commas=True, unit="", kind="number"):
    """
    Format an inclusive range ``lo–hi`` using an **en-dash** (MIT Press style).

    MIT Press house style for ranges (style sheet §ranges): use an en-dash,
    never a hyphen, and write **both endpoints in full** — ``1992–1993`` not
    ``1992–93``, ``5–10`` not ``5-10``. This helper owns the en-dash so authors
    never type ``-`` or ``--`` between two values, and applies the chosen
    value-kind formatter to each endpoint so the unit/symbol is consistent.

        fmt_range(5, 10, precision=0)                       # "5–10"
        fmt_range(5, 10, precision=0, unit="GB")            # "5–10 GB"
        fmt_range(1992, 1993, precision=0, commas=False)    # "1992–1993"
        fmt_range(0.10, 0.50, kind="usd", precision=2)      # "\\$0.10–\\$0.50"
        fmt_range(2, 4, precision=0, unit="percent")        # "2–4 percent"

    ``kind`` selects the per-endpoint formatter: ``"number"`` (default, via
    ``fmt``) or ``"usd"`` (via ``fmt_usd`` — each endpoint carries ``\\$``). For
    a percent range, format the display scalars as numbers with
    ``unit="percent"`` (body) or ``unit="%"`` (table).

    Guards: both endpoints finite; ``hi >= lo`` (an inverted range is almost
    always a bug); precision/sign guards inherited from the endpoint formatter.
    """
    lo_v = _numeric_magnitude(lo)
    hi_v = _numeric_magnitude(hi)
    if hi_v < lo_v:
        raise ValueError(
            f"fmt_range expects hi >= lo, got lo={lo_v}, hi={hi_v}. An inverted "
            f"range is almost always a bug; swap the endpoints at the source."
        )
    if kind == "usd":
        a = fmt_usd(lo, precision=precision, commas=commas)
        b = fmt_usd(hi, precision=precision, commas=commas)
    elif kind == "number":
        a = fmt(lo, precision=precision, commas=commas)
        b = fmt(hi, precision=precision, commas=commas)
    else:
        raise ValueError(
            f"fmt_range kind must be 'number' or 'usd', got {kind!r}. For a "
            f"percent range, use kind='number' with unit='percent'/'%'."
        )
    tail = f" {unit}" if unit else ""
    return MarkdownStr(f"{a}\u2013{b}{tail}")


def fmt_sci(val, precision=2):
    """
    Formats a number or Pint Quantity into scientific notation using Unicode.
    Example: 4.1e9 -> "4.10 × 10⁹"
    """
    # Unicode superscript digits
    superscripts = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻",
    }

    if isinstance(val, ureg.Quantity):
        val = val.magnitude
    _require_finite(float(val), "scientific value")
    s = f"{val:.{precision}e}"
    base, exp = s.split("e")
    exp_int = int(exp)
    exp_str = "".join(superscripts.get(c, c) for c in str(exp_int))
    return MarkdownStr(f"{float(base):.{precision}f} × 10{exp_str}")


def sci_latex(val, precision=2):
    """
    Formats a number or Pint Quantity into LaTeX scientific notation.
    Example: 4.1e9 -> "4.10 \\times 10^{9}"
    """
    if isinstance(val, ureg.Quantity):
        val = val.magnitude
    _require_finite(float(val), "scientific value")
    s = f"{val:.{precision}e}"
    base, exp = s.split('e')
    exp_int = int(exp)
    return f"{float(base):.{precision}f} \\times 10^{{{exp_int}}}"


def fmt_frac(numerator, denominator, result=None, unit=None):
    """
    Create a LaTeX fraction with optional result and unit.
    Returns: $\\frac{num}{denom}$ or $\\frac{num}{denom} = result$ unit
    """
    latex = f'$\\frac{{{numerator}}}{{{denominator}}}$'
    if result is not None:
        latex += f' = {result}'
    if unit is not None:
        latex += f' {unit}'
    out = MarkdownStr(latex)
    assert isinstance(out, MarkdownStr), "fmt_frac() must return MarkdownStr"
    return out


def _compact_unit_suffix(display_unit) -> str:
    """Derive a leading-space compact unit label from a pint display unit."""
    # Currency is not a fmt_qty value-kind. It must go through fmt_usd(), which
    # owns the Pandoc-safe escaped "\$" and never emits the literal "USD".
    # Routing dollars through fmt_qty here would print a bare " USD" suffix that
    # the source currency checker cannot see (it is generated at render time).
    if str(display_unit) in {"dollar", "USD", "EUR"}:
        raise ValueError(
            "fmt_qty() does not format currency. Use fmt_usd(amount, ...) so the "
            "dollar sign is escaped for prose and no literal 'USD' is emitted. "
            "See .claude/rules/numbers-and-math-in-prose.md §4."
        )
    try:
        one = 1 * display_unit
        formatted = f"{one:~P}"
    except Exception:
        return f" {display_unit}"
    parts = formatted.split(None, 1)
    if len(parts) == 2:
        return f" {parts[1]}"
    return f" {display_unit}"


def fmt_qty(
    quantity,
    display_unit,
    *,
    precision=1,
    commas=False,
    prefix="",
    extra_suffix="",
    approx=False,
    lower_bound=False,
):
    """Format a pint Quantity in ``display_unit`` with a canonical unit suffix.

    Required OUTPUT path for physical quantities in LEGO cells.
    The value must remain a Pint Quantity until this function so the formatter
    can dimension-check the conversion before delegating to ``fmt``.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError(
            "fmt_qty() requires a Pint Quantity. Keep units attached at the "
            "call site, e.g. fmt_qty(bw, GB/second), not "
            "fmt_qty(bw.m_as(GB/second), GB/second)."
        )
    q = quantity.to(display_unit)
    val = q.magnitude
    suffix = _compact_unit_suffix(display_unit) + extra_suffix
    return fmt(
        val,
        precision=precision,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


def check(condition, message):
    """
    Invariant guard for narrative logic.
    Ensures that the calculated values support the textbook's claims.
    """
    if not condition:
        raise ValueError(f"Narrative broken: {message}")


def fmt_math(expression):
    """
    Wrap a LaTeX math expression in `$...$` so Quarto renders it as inline math.
    Returns a MarkdownStr.
    """
    out = MarkdownStr(f"${expression}$")
    assert isinstance(out, MarkdownStr), "fmt_math() must return MarkdownStr"
    return out
