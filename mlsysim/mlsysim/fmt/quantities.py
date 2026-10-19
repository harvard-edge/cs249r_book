"""Generic Pint Quantity, scientific notation, fraction, and time formatters."""

from ..core.units import ureg
from .core import (
    MarkdownStr,
    _clean_text_atom,
    _compact_unit_suffix,
    _denominator_suffix,
    _numeric_magnitude,
    _require_finite,
    _resolve_display_precision,
    _resolve_unit_arg,
    fmt,
    fmt_int,
)


def fmt_sci(val, precision=2):
    """
    Formats a number or Pint Quantity into scientific notation using Unicode.
    Example: 4.1e9 -> "4.10 × 10⁹"
    """
    # Unicode superscript digits
    superscripts = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
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
    base, exp = s.split("e")
    exp_int = int(exp)
    return f"{float(base):.{precision}f} \\times 10^{{{exp_int}}}"


def fmt_frac(numerator, denominator, result=None, unit=None):
    """
    Create a LaTeX fraction with optional result and unit.
    Returns: $\\frac{num}{denom}$ or $\\frac{num}{denom} = result$ unit
    """
    latex = f"$\\frac{{{numerator}}}{{{denominator}}}$"
    if result is not None:
        latex += f" = {result}"
    if unit is not None:
        latex += f" {unit}"
    out = MarkdownStr(latex)
    assert isinstance(out, MarkdownStr), "fmt_frac() must return MarkdownStr"
    return out


def _quantity_suffix(display_unit, *, unit_label=None, per=None, extra_suffix=""):
    """Build the checked unit suffix shared by quantity formatters."""
    if extra_suffix and per is not None:
        raise ValueError("Use extra_suffix= or structured per=, not both.")
    unit_suffix = _compact_unit_suffix(display_unit)
    if unit_label is not None:
        unit_suffix = f" {_clean_text_atom(unit_label, what='fmt_qty unit_label')}"
    return unit_suffix + _denominator_suffix(per, what="per") + extra_suffix


def fmt_qty(
    quantity,
    display_unit=None,
    *,
    unit=None,
    precision=1,
    commas=False,
    prefix="",
    extra_suffix="",
    unit_label=None,
    per=None,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    trim_trailing_zeros=False,
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
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_qty")
    q = quantity.to(display_unit)
    val = q.magnitude
    suffix = _quantity_suffix(
        display_unit,
        unit_label=unit_label,
        per=per,
        extra_suffix=extra_suffix,
    )
    return fmt(
        val,
        precision=precision,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=trim_trailing_zeros,
    )


def fmt_magnitude(
    quantity,
    display_unit=None,
    *,
    unit=None,
    precision=1,
    commas=False,
    prefix="",
    suffix="",
    approx=False,
    lower_bound=False,
):
    """Format a Pint quantity's magnitude in an explicit unit, without a suffix.

    This is the sanctioned scalar boundary for algebraic displays where the
    surrounding equation, table header, or variable name carries the unit.
    It keeps the Pint conversion inside ``mlsysim.fmt`` while intentionally
    rendering only the numeric magnitude. For prose-facing closed outputs,
    prefer ``fmt_qty`` or a domain formatter so the unit is rendered with the
    value.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError(
            "fmt_magnitude() requires a Pint Quantity. Use fmt()/fmt_count() "
            "only for values that are already intentionally dimensionless."
        )
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_magnitude")
    q = quantity.to(display_unit)
    return fmt(
        q.magnitude,
        precision=precision,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


def fmt_qty_int(
    quantity,
    display_unit=None,
    *,
    unit=None,
    commas=True,
    prefix="",
    extra_suffix="",
    unit_label=None,
    per=None,
    approx=False,
    lower_bound=False,
):
    """Format a Pint Quantity as a rounded integer with a checked unit suffix.

    Use this only when rounded integer display is the editorial intent. For
    exact integer quantities, prefer ``fmt_qty(..., precision=0)`` because it
    rejects fractional values instead of rounding them.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError(
            "fmt_qty_int() requires a Pint Quantity. Keep units attached at "
            "the call site, e.g. fmt_qty_int(memory, GB), not "
            "fmt_qty_int(memory.m_as(GB), GB)."
        )
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_qty_int")
    q = quantity.to(display_unit)
    suffix = _quantity_suffix(
        display_unit,
        unit_label=unit_label,
        per=per,
        extra_suffix=extra_suffix,
    )
    return fmt_int(
        q.magnitude,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


_TIME_WORDS = {
    "nanosecond": ("nanosecond", "nanoseconds"),
    "ns": ("nanosecond", "nanoseconds"),
    "NS": ("nanosecond", "nanoseconds"),
    "microsecond": ("microsecond", "microseconds"),
    "us": ("microsecond", "microseconds"),
    "µs": ("microsecond", "microseconds"),
    "μs": ("microsecond", "microseconds"),
    "US": ("microsecond", "microseconds"),
    "millisecond": ("millisecond", "milliseconds"),
    "ms": ("millisecond", "milliseconds"),
    "MS": ("millisecond", "milliseconds"),
    "second": ("second", "seconds"),
    "s": ("second", "seconds"),
    "minute": ("minute", "minutes"),
    "min": ("minute", "minutes"),
    "hour": ("hour", "hours"),
    "h": ("hour", "hours"),
    "day": ("day", "days"),
    "d": ("day", "days"),
    "week": ("week", "weeks"),
    "month": ("month", "months"),
    "year": ("year", "years"),
    "a": ("year", "years"),
}


_TIME_MARKERS = {"+": "trailing plus marker"}


def _time_word_suffix(value, display_unit, per=None) -> str:
    """Return a leading-space time-unit word suffix with plural agreement."""
    key = str(display_unit)
    if key not in _TIME_WORDS:
        raise ValueError(f"fmt_time style='word' does not know a word label for {key!r}.")
    singular, plural = _TIME_WORDS[key]
    word = singular if abs(value - 1) <= 1e-9 else plural
    return f" {word}{_denominator_suffix(per, what='per')}"


def _time_attributive_suffix(display_unit) -> str:
    """Return a hyphenated singular time-unit suffix for noun modifiers."""
    key = str(display_unit)
    if key not in _TIME_WORDS:
        raise ValueError(f"fmt_time attributive=True does not know a word label for {key!r}.")
    singular, _ = _TIME_WORDS[key]
    return f"-{singular}"


def fmt_time(
    duration,
    display_unit=None,
    *,
    unit=None,
    precision=1,
    commas=False,
    style="symbol",
    attributive=False,
    per=None,
    marker="",
    approx=False,
    lower_bound=False,
    allow_negative=False,
    trim_trailing_zeros=False,
):
    """Format a duration with time-specific checks and defaults.

    ``fmt_qty`` remains the generic physical-quantity formatter. ``fmt_time``
    exists because durations are common enough to deserve stricter semantics:
    the display unit must be a Pint time unit, values are non-negative by
    default, and comma grouping defaults off.

    ``style="symbol"`` renders compact unit symbols such as ``35 ms``.
    ``style="word"`` renders prose words with singular/plural agreement such
    as ``1 second`` and ``2 seconds``.
    ``attributive=True`` with ``style="word"`` renders a hyphenated singular
    noun modifier such as ``10-minute`` or ``100,000-hour``.
    ``marker="+"`` appends a checked trailing plus to compact values such as
    ``100 ms+``.

    Plain numbers are accepted only because the display unit is explicit:
    ``fmt_time(35, second)`` means "35 seconds" and still validates that
    ``second`` is a time unit.
    """
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_time")
    one = 1 * display_unit
    if not one.check("[time]"):
        raise ValueError(f"fmt_time display_unit must be a time unit, got {display_unit}.")
    if style not in {"symbol", "word"}:
        raise ValueError(f"fmt_time style must be 'symbol' or 'word', got {style!r}.")
    if attributive and style != "word":
        raise ValueError("fmt_time attributive=True requires style='word'.")
    if attributive and per is not None:
        raise ValueError("fmt_time attributive=True cannot be combined with per=.")
    if marker and marker not in _TIME_MARKERS:
        raise ValueError(f"fmt_time marker must be one of {sorted(_TIME_MARKERS)}, got {marker!r}.")
    if marker and style != "symbol":
        raise ValueError("fmt_time marker= is only supported with style='symbol'.")
    if marker and (attributive or per is not None):
        raise ValueError("fmt_time marker= cannot be combined with attributive=True or per=.")
    q = duration if isinstance(duration, ureg.Quantity) else duration * display_unit
    val = _numeric_magnitude(q.to(display_unit))
    auto_precision = precision is None
    p = _resolve_display_precision(val, precision)
    trim = trim_trailing_zeros or auto_precision
    if val < 0 and not allow_negative:
        raise ValueError(
            f"fmt_time expects a non-negative duration, got {val}. Pass "
            f"allow_negative=True if a signed duration is genuinely intended."
        )
    if attributive:
        return fmt(
            val,
            precision=p,
            commas=commas,
            suffix=_time_attributive_suffix(display_unit),
            approx=approx,
            lower_bound=lower_bound,
            trim_trailing_zeros=trim,
        )
    if style == "word":
        return fmt(
            val,
            precision=p,
            commas=commas,
            suffix=_time_word_suffix(val, display_unit, per),
            approx=approx,
            lower_bound=lower_bound,
            trim_trailing_zeros=trim,
        )
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        per=per,
        extra_suffix=marker,
        approx=approx,
        lower_bound=lower_bound,
        trim_trailing_zeros=trim,
    )
