"""
formatting.py
Formatting + presentation helpers for QMD output.
Keep science in formulas.py; keep display here.
"""

from .constants import ureg

# Lazy import for IPython.display.Markdown
_Markdown = None


def _get_markdown():
    """Lazily import IPython.display.Markdown when first needed."""
    global _Markdown
    if _Markdown is None:
        from IPython.display import Markdown
        _Markdown = Markdown
    return _Markdown


def fmt(quantity, unit=None, precision=1, commas=True, allow_zero=False):
    """
    Format a Pint Quantity for narrative text.
    Returns ONLY the number string (no unit suffix).

    Safety: Raises ValueError if a non-zero value is formatted as "0"
    due to insufficient precision (unless allow_zero=True).
    """
    if unit:
        # If a raw number is passed, assume it is already in base units.
        if isinstance(quantity, ureg.Quantity):
            quantity = quantity.to(unit)

    if isinstance(quantity, ureg.Quantity):
        val = quantity.magnitude
    else:
        val = quantity

    # Primary formatting
    fmt_str = f",.{precision}f" if commas else f".{precision}f"
    result = f"{val:{fmt_str}}"

    # --- Precision Safety Check ---
    # Check if we accidentally rounded a non-zero value to zero
    try:
        numeric_result = float(result.replace(",", ""))
    except ValueError:
        numeric_result = None # Case for non-numeric strings if any

    if numeric_result == 0.0 and abs(val) > 1e-12 and not allow_zero:
        raise ValueError(
            f"Formatting Precision Error: Value {val} was formatted as '{result}' "
            f"with precision={precision}. This hides the actual value. "
            f"Increase precision or set allow_zero=True if this was intentional."
        )

    return result


def fmt_percent(ratio, precision=1, commas=False):
    """
    Format a ratio (0.0 to 1.0) as a percentage string for display.
    Use this for compound fractions (e.g. effective utilization) to avoid
    display bugs from Quantity or wrong scaling.
    Accepts Pint Quantity (uses magnitude) or plain float.
    """
    if isinstance(ratio, ureg.Quantity):
        # Crucial: convert to dimensionless first so units like flop/TFLOP cancel out!
        ratio = float(ratio.m_as(''))
    else:
        ratio = float(ratio)
    return fmt(ratio * 100, precision=precision, commas=commas)


def sci(val, precision=2):
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
    return f"{float(base):.{precision}f} × 10{exp_str}"


def display_value(value, unit=None, precision=1, commas=True):
    """
    Return a dict with raw value and formatted string.
    """
    return {
        "value": value,
        "str": fmt(value, unit=unit, precision=precision, commas=commas),
    }


def display_percent(ratio, precision=0):
    """
    ratio: 0.0 to 1.0
    """
    if isinstance(ratio, ureg.Quantity):
        ratio = float(ratio.m_as(''))
    else:
        ratio = float(ratio)
        
    pct = ratio * 100
    return {
        "value": ratio,
        "str": f"{pct:.{precision}f}",
        "math": md_math(f"{pct:.{precision}f}\\%"),
    }


def display_fraction(numerator, denominator, result=None, unit=None, precision=2):
    """
    numerator/denominator can be numeric or Pint Quantity.
    result is optional string; if omitted, it will be computed.
    """
    if result is None:
        result = f"{(numerator / denominator):.{precision}g}"
    num_latex = sci_latex(numerator, precision=precision)
    den_latex = sci_latex(denominator, precision=precision)
    return {
        "value": numerator / denominator,
        "str": result,
        "frac": md_frac(num_latex, den_latex, result=result, unit=unit),
    }


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


def md(latex_str):
    """
    Wrap a LaTeX string in Markdown() to preserve formatting in inline code.
    """
    Markdown = _get_markdown()
    return Markdown(latex_str)


def md_frac(numerator, denominator, result=None, unit=None):
    """
    Create a LaTeX fraction with optional result and unit.
    Returns: $\frac{num}{denom}$ or $\frac{num}{denom} = result$ unit
    """
    Markdown = _get_markdown()
    latex = f'$\\frac{{{numerator}}}{{{denominator}}}$'
    if result is not None:
        latex += f' = {result}'
    if unit is not None:
        latex += f' {unit}'
    return Markdown(latex)


def md_sci(val, precision=2):
    """
    Format a number in LaTeX scientific notation, wrapped in Markdown().
    """
    Markdown = _get_markdown()
    return Markdown(f"${sci_latex(val, precision=precision)}$")


def check(condition, message):
    """
    Invariant guard for narrative logic.
    Ensures that the calculated values support the textbook's claims.
    """
    if not condition:
        raise ValueError(f"Narrative broken: {message}")


def md_math(expression):
    """
    Wrap a LaTeX math expression in Markdown().
    """
    return md(f"${expression}$")


def fmt_full(quantity, precision=1, commas=True):
    """
    Format a Pint Quantity as a complete "value unit" string.
    Returns a single string like "2,039 GB/s" or "312 TFLOPs/s".

    Value and unit are always in sync — unit is taken directly from the Quantity.

    RULE: Use the whole string in prose. Do NOT add a separate unit label.
        ✓ "`{python} bw_str` of memory bandwidth"
        ✗ "`{python} bw_str` GB/s"  ← unit already in string; this doubles it

    Use fmt()  when the unit is fixed and hardcoded in prose.
    Use fmt_split() for tables needing separate value/unit columns.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError(
            f"fmt_full() requires a pint Quantity, got {type(quantity).__name__}. "
            f"Use fmt() for raw numbers."
        )
    val = quantity.magnitude
    unit_str = f"{quantity.units:~P}"   # e.g. "GB/s", "TFLOPs/s"
    fmt_str = f",.{precision}f" if commas else f".{precision}f"
    value_str = f"{val:{fmt_str}}"

    # Precision safety check (same guard as fmt())
    try:
        numeric_result = float(value_str.replace(",", ""))
    except ValueError:
        numeric_result = None
    if numeric_result == 0.0 and abs(val) > 1e-12:
        raise ValueError(
            f"fmt_full() Precision Error: {val} formatted as '{value_str}' "
            f"with precision={precision}. Increase precision."
        )
    return f"{value_str} {unit_str}"


def fmt_split(quantity, precision=1, commas=True):
    """
    Format a Pint Quantity as a (value_str, unit_str) tuple. For table columns only.
    For prose, use fmt_full() instead.

        bw_val, bw_unit = fmt_split(A100_MEM_BW)  # ("2,039", "GB/s")
    """
    full = fmt_full(quantity, precision=precision, commas=commas)
    parts = full.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (full, "")
