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


def _numeric_magnitude(quantity):
    """Return a plain float magnitude for fmt safety checks."""
    from .core.provenance import Sourced, scalar_value

    if isinstance(quantity, ureg.Quantity):
        return float(quantity.magnitude)
    if isinstance(quantity, Sourced):
        return scalar_value(quantity)
    return float(quantity)


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


def fmt(quantity, unit=None, precision=1, commas=True,
        prefix="", suffix=""):
    """
    Format a Pint Quantity (or plain number) for narrative text.
    Returns a MarkdownStr so Quarto inserts the value verbatim (no escape).

    The prefix and suffix arguments collapse the old MarkdownStr(f"...")
    escape-hatch idiom into a single canonical helper. Common uses:

        fmt(price, precision=0, prefix="$")         # "$1,000"
        fmt(rate * 100, precision=1, commas=False, suffix="%")  # "12.4%"
        fmt(bw_mb_s, precision=1, commas=False, suffix=" MB/s")  # "2.4 MB/s"
        fmt(speedup, precision=0, commas=False, suffix="x")     # "8x"

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

    decorated = f"{prefix}{result}{suffix}"
    out = MarkdownStr(decorated)
    assert isinstance(out, MarkdownStr), (
        "fmt() must return MarkdownStr — this guard exists so a future refactor "
        "of this module cannot silently break Quarto's _repr_markdown_ detection. "
        "See the project math rules."
    )
    return out


def fmt_int(quantity, unit=None, commas=True, prefix="", suffix=""):
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
    return fmt(round(val), precision=0, commas=commas, prefix=prefix, suffix=suffix)


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


def fmt_percent(ratio, precision=1, commas=False, suffix=""):
    """
    Format a ratio (0.0 to 1.0) as a percentage string for display.

    Use this for compound fractions (e.g. effective utilization) to avoid
    display bugs from Quantity or wrong scaling.
    Accepts Pint Quantity (uses magnitude) or plain float.

    suffix=" percent" for body prose (MIT Press: spell out "percent").
    suffix="%" for tables, equations, and constrained captions.
    No suffix (default) when the prose context carries the meaning
    (e.g., "50 MFU", "85 goodput").
    """
    if isinstance(ratio, ureg.Quantity):
        ratio = float(ratio.m_as(''))
    else:
        ratio = float(ratio)
    return fmt(ratio * 100, precision=precision, commas=commas, suffix=suffix)


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
    from .core.units import USD

    if display_unit is USD:
        return " USD"
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
):
    """Format a pint Quantity in ``display_unit`` with a canonical unit suffix.

    Required OUTPUT path for physical quantities in LEGO cells.
    """
    if isinstance(quantity, ureg.Quantity):
        q = quantity.to(display_unit)
        val = q.magnitude
    else:
        val = _numeric_magnitude(quantity)
    suffix = _compact_unit_suffix(display_unit) + extra_suffix
    return fmt(val, precision=precision, commas=commas, prefix=prefix, suffix=suffix)


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
