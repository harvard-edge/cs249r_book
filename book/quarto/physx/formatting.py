"""
formatting.py
Formatting + presentation helpers for QMD output.
Keep science in formulas.py; keep display here.
"""

# Lazy import for IPython.display.Markdown
_Markdown = None


def _get_markdown():
    """Lazily import IPython.display.Markdown when first needed."""
    global _Markdown
    if _Markdown is None:
        from IPython.display import Markdown
        _Markdown = Markdown
    return _Markdown


def fmt(quantity, unit=None, precision=1, commas=True):
    """
    Format a Pint Quantity for narrative text.
    Returns ONLY the number string (no unit suffix).
    """
    if unit:
        # If a raw number is passed, assume it is already in base units.
        if hasattr(quantity, "magnitude"):
            quantity = quantity.to(unit)

    if hasattr(quantity, "magnitude"):
        val = quantity.magnitude
    else:
        val = quantity

    fmt_str = f",.{precision}f" if commas else f".{precision}f"
    return f"{val:{fmt_str}}"


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

    if hasattr(val, "magnitude"):
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
    if hasattr(val, "magnitude"):
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


def md_math(expression):
    """
    Wrap a LaTeX math expression in Markdown().
    """
    return md(f"${expression}$")
