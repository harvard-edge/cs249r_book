"""
fmt.py
Formatting + presentation helpers for Markdown/Quarto output.
Keep science in mlsysim/physics/; keep display here.
"""

from .core.units import ureg


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


def _dimensionless_magnitude(value, *, what):
    """Return a scalar magnitude only after checking dimensional consistency."""
    if isinstance(value, ureg.Quantity):
        if value.units != ureg.dimensionless:
            raise ValueError(
                f"{what} expects a dimensionless scalar, got units {value.units}. "
                "Convert same-unit quotients to dimensionless at the source, or "
                "use a domain formatter for unit-bearing quantities."
            )
        return _require_finite(float(value.to("").magnitude), what)
    return _numeric_magnitude(value)


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
    # Non-numeric render (e.g. exotic format output) — nothing to check.
    numeric_result = _parse_formatted_number(result)
    if numeric_result is None:
        return

    # Mode 1 — vanishing value: the rounded display is exactly zero while the
    # input is not. This is the original MarkdownStr-era bug class (0.001
    # rendered as "0"), and it fires at ANY precision.
    if numeric_result == 0.0 and abs(val) > 1e-12:
        raise ValueError(
            f"Formatting Precision Error: Value {val} was formatted as '{result}' "
            f"with precision={precision}. This hides the actual value. "
            f"Increase precision or change units to avoid representing a "
            f"non-zero value as zero."
        )

    # Mode 2 — silent integer rounding: precision=0 on a genuinely fractional
    # value (10.7 -> "11") corrupts arithmetic quoted in prose without any
    # visible symptom. The 1e-9 tolerance forgives float noise on true ints.
    if precision == 0 and abs(val) > 1e-12:
        nearest_int = round(val)
        if abs(val - nearest_int) > 1e-9:
            raise ValueError(
                f"Formatting Precision Error: Value {val} is not integer-like "
                f"but precision=0 formats it as '{result}'. "
                f"Use precision>=1 to preserve the fractional part, or "
                f"fmt_int({val!r}) if integer display is intentional."
            )

    # Mode 3 — the cosmetic inverse: an integer-like value dressed with dead
    # decimals ("512.0"). Not a correctness bug, but a house-style defect the
    # guard surfaces at the call site rather than in copyedit.
    if precision >= 1 and _is_integer_like(val) and _has_spurious_zero_decimals(result):
        raise ValueError(
            f"Formatting Precision Error: Value {val} is integer-like "
            f"but precision={precision} formats it as '{result}'. "
            f"Use precision=0 or fmt_int(...) to avoid spurious trailing zeros."
        )


def _display_prefix(prefix="", *, approx=False, lower_bound=False, upper_bound=False):
    """Build a standardized leading marker for a formatted display value."""
    active = [approx, lower_bound, upper_bound]
    if sum(bool(x) for x in active) > 1:
        raise ValueError("A value can use only one display marker: approx, lower_bound, or upper_bound.")
    marker = "~" if approx else ("> " if lower_bound else ("< " if upper_bound else ""))
    if prefix and marker:
        raise ValueError("Use either prefix= or a named display marker, not both.")
    return prefix or marker


def _clean_text_atom(value, *, what):
    """Validate a structured display atom such as a count label or denominator."""
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}.")
    if not value:
        raise ValueError(f"{what} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{what} must not have leading/trailing whitespace.")
    forbidden = {"$", "\\$", "%", "×"}
    if any(tok in value for tok in forbidden):
        raise ValueError(f"{what} must not contain currency, percent, or multiplier glyphs.")
    return value


_COUNT_LABEL_DENYLIST = {
    "GB",
    "MB",
    "KB",
    "TB",
    "PB",
    "GiB",
    "MiB",
    "KiB",
    "TiB",
    "GB/s",
    "MB/s",
    "TB/s",
    "Gb/s",
    "TFLOP/s",
    "PFLOP/s",
    "W",
    "kW",
    "MW",
    "J",
    "mJ",
    "Wh",
    "kWh",
    "MWh",
    "ms",
    "s",
    "min",
    "h",
    "ns",
    "us",
    "µs",
    "μs",
    "QPS",
    "FPS",
    "tokens/s",
    "tokens/hour",
    "img/s",
    "images/s",
    "req/s",
    "samples/s",
    "percent",
    "percentage point",
    "percentage points",
}


_DECIMAL_SCALE_FACTORS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_DECIMAL_SCALE_WORDS = {
    "K": "thousand",
    "M": "million",
    "B": "billion",
    "T": "trillion",
}
_DECIMAL_SCALE_SYMBOL_BY_WORD = {word: symbol for symbol, word in _DECIMAL_SCALE_WORDS.items()}


def _resolve_decimal_scale(scale, *, what, style="symbol", attributive=False):
    """Return ``(divisor, suffix)`` for checked K/M/B/T decimal scales."""
    if style not in {"symbol", "word"}:
        raise ValueError(f"{what} style must be 'symbol' or 'word', got {style!r}.")
    if scale is None:
        if attributive:
            raise ValueError(f"{what} attributive=True requires scale=.")
        return 1, ""
    if not isinstance(scale, str):
        raise TypeError(f"{what} must be a string or None, got {type(scale).__name__}.")
    if scale in _DECIMAL_SCALE_FACTORS:
        if attributive and style != "word":
            raise ValueError(f"{what} attributive=True requires a word scale, got {scale!r}.")
        divisor = _DECIMAL_SCALE_FACTORS[scale]
        if style == "word":
            joiner = "-" if attributive else " "
            return divisor, f"{joiner}{_DECIMAL_SCALE_WORDS[scale]}"
        return divisor, scale
    word_key = scale.lower()
    if word_key in _DECIMAL_SCALE_SYMBOL_BY_WORD:
        symbol = _DECIMAL_SCALE_SYMBOL_BY_WORD[word_key]
        joiner = "-" if attributive else " "
        return _DECIMAL_SCALE_FACTORS[symbol], f"{joiner}{scale}"
    allowed = sorted(_DECIMAL_SCALE_FACTORS) + sorted(_DECIMAL_SCALE_SYMBOL_BY_WORD)
    raise ValueError(f"{what} must be one of {allowed}, got {scale!r}.")


def _auto_decimal_scale(raw_value):
    """Pick the largest K/M/B/T scale that keeps a count display >= 1."""
    abs_value = abs(raw_value)
    for symbol, divisor in sorted(
        _DECIMAL_SCALE_FACTORS.items(), key=lambda item: item[1], reverse=True
    ):
        if abs_value >= divisor:
            return symbol
    return None


def _resolve_scaled_count_precision(val, precision):
    """Pick precision for scaled counts without hiding sub-unit explicit scales."""
    if precision is not None:
        return precision
    if _is_integer_like(val):
        return 0
    # Sub-unit mantissas arise when an explicit scale exceeds the value (e.g.
    # 150M rendered at scale="B" → 0.15). Step precision up one decimal per
    # decade below 1 so at least two significant digits survive — otherwise
    # the precision guard would (correctly) reject the display as "0.0".
    abs_val = abs(val)
    if abs_val >= 1:
        return _resolve_display_precision(val, None)
    if abs_val >= 0.1:
        return 2
    if abs_val >= 0.01:
        return 3
    return 4


def _validate_count_label(label, *, what="label") -> str:
    """Validate a count noun label before pluralization."""
    label = _clean_text_atom(label, what=what)
    if "/" in label:
        raise ValueError(f"{what} must be a count noun, not a rate/unit expression.")
    if label in _COUNT_LABEL_DENYLIST:
        raise ValueError(
            f"{what}={label!r} looks like a unit, rate, percent, or glyph label. "
            "Use the formatter for that value kind instead."
        )
    return label


def _pluralize_label(label: str) -> str:
    """Best-effort English plural for count labels."""
    if label.endswith("y") and len(label) > 1 and label[-2].lower() not in "aeiou":
        return f"{label[:-1]}ies"
    if label.endswith(("s", "x", "ch", "sh")):
        return f"{label}es"
    return f"{label}s"


def _label_suffix(raw_value, label=None, plural_label=None) -> str:
    """Return a leading-space count label that agrees with the raw count."""
    if label is None:
        if plural_label is not None:
            raise ValueError("plural_label requires label.")
        return ""
    label = _validate_count_label(label, what="label")
    plural = (
        _validate_count_label(plural_label, what="plural_label")
        if plural_label is not None
        else _pluralize_label(label)
    )
    word = label if abs(raw_value - 1) <= 1e-9 else plural
    return f" {word}"


def _validate_count_value(
    raw_value,
    *,
    allow_fractional=False,
    require_integer=True,
):
    """Reject negative and, when requested, fractional raw counts."""
    if raw_value < 0:
        raise ValueError(f"fmt_count expects a non-negative count, got {raw_value}.")
    if require_integer and not allow_fractional and not _is_integer_like(raw_value):
        raise ValueError(
            f"fmt_count expects a whole-number count, got {raw_value}. Pass "
            "allow_fractional=True only when the count is intentionally "
            "fractional."
        )


_UNIT_LABEL_NORMALIZATION = {
    "KFLOPs": "KFLOP",
    "MFLOPs": "MFLOP",
    "GFLOPs": "GFLOP",
    "TFLOPs": "TFLOP",
    "PFLOPs": "PFLOP",
    "EFLOPs": "EFLOP",
    "ZFLOPs": "ZFLOP",
    "KFLOPs/s": "KFLOP/s",
    "MFLOPs/s": "MFLOP/s",
    "GFLOPs/s": "GFLOP/s",
    "TFLOPs/s": "TFLOP/s",
    "PFLOPs/s": "PFLOP/s",
    "EFLOPs/s": "EFLOP/s",
    "ZFLOPs/s": "ZFLOP/s",
    "Gbps": "Gb/s",
    "Mbit/s": "Mb/s",
    "Gbit/s": "Gb/s",
    "Tbit/s": "Tb/s",
    "B": "bytes",
    "flop/B": "FLOP/byte",
    "pJ/B": "pJ/byte",
    "pJ/flop": "pJ/FLOP",
    "µs": "μs",
}


def _normalize_unit_label(label: str) -> str:
    """Map a pint compact unit label onto the book's canonical display label.

    Single source of truth shared by ``fmt_unit`` and ``fmt_qty``'s suffix
    builder (2026-06: fmt_unit previously carried a divergent local map that
    missed EFLOPs/s, KFLOPs/s, and bare work units such as TFLOPs).
    """
    return _UNIT_LABEL_NORMALIZATION.get(label, label.replace("µs", "μs"))


def _coerce_unit(display_unit):
    """Return a Pint Unit from a Pint unit-like object or unit string."""
    if isinstance(display_unit, str):
        return ureg.Unit(display_unit)
    return display_unit


def _resolve_unit_arg(display_unit, unit, *, what):
    """Resolve the display unit from positional or keyword ``unit=`` spelling."""
    if display_unit is None and unit is None:
        raise TypeError(f"{what} requires a display unit.")
    if display_unit is not None and unit is not None:
        raise TypeError(f"{what} accepts either a positional unit or unit=, not both.")
    return _coerce_unit(unit if unit is not None else display_unit)


_USD_DENOMINATORS = {
    "month",
    "year",
    "hour",
    "hr",
    "day",
    "week",
    "GB",
    "TB",
    "GB/month",
    "TB/month",
    "kWh",
    "MWh",
    "(TFLOP/s)",
    "label",
    "image",
    "query",
    "request",
    "token",
    "inference",
    "sample",
    "click",
    "run",
    "million queries",
    "million tokens",
    "1K inferences",
    "million",
    "tonne",
    "device",
    "device/year",
    "GPU-hour",
}


def _denominator_suffix(per, *, what="per", allowed=None) -> str:
    """Return a structured denominator suffix such as /month or /GB."""
    if per is None:
        return ""
    if isinstance(per, str):
        value = _clean_text_atom(per, what=what)
        if value.startswith("/"):
            raise ValueError(f"{what} should omit the leading '/', got {per!r}.")
        if allowed is not None and value not in allowed:
            raise ValueError(f"{what} must be one of {sorted(allowed)}, got {value!r}.")
        return f"/{value}"
    unit_label = _compact_unit_suffix(_coerce_unit(per)).strip()
    if allowed is not None and unit_label not in allowed:
        raise ValueError(f"{what} must be one of {sorted(allowed)}, got {unit_label!r}.")
    return f"/{unit_label}"


def _usd_denominator_unit(per):
    """Return the Pint denominator unit implied by ``fmt_usd(per=...)``.

    Display strings such as ``"TB/month"`` mean dollars per TB per month, so
    their physical denominator is the product ``TB * month``. Parenthesized
    strings such as ``"(TFLOP/s)"`` are intentionally parsed as one compound
    unit because the prose meaning is dollars per throughput.
    """
    if per is None:
        return None
    if not isinstance(per, str):
        return _coerce_unit(per)
    value = _clean_text_atom(per, what="fmt_usd per")
    if value.startswith("/"):
        raise ValueError(f"fmt_usd per should omit the leading '/', got {per!r}.")
    if value.startswith("(") and value.endswith(")"):
        return ureg.Unit(value[1:-1])
    denom = None
    for part in value.split("/"):
        unit = ureg.Unit(part)
        denom = unit if denom is None else denom * unit
    return denom


def _usd_amount_magnitude(amount, per):
    """Coerce a scalar or Pint dollar Quantity into fmt_usd's display domain."""
    from .core.units import USD

    if not isinstance(amount, ureg.Quantity):
        return amount

    unit_label = f"{amount.units}"
    if "dollar" not in unit_label:
        raise ValueError(
            f"fmt_usd expects a dollar Quantity, got units {amount.units}. "
            "Pass a plain numeric dollar magnitude only at an intentional "
            "scalar boundary."
        )

    if per is None:
        if unit_label != "dollar":
            raise ValueError(
                f"fmt_usd received a dollar rate with units {amount.units} but "
                "no per= denominator. Pass per= so the formatter owns the "
                "unit conversion."
            )
        return amount.m_as(USD)

    try:
        denominator = _usd_denominator_unit(per)
    except Exception:
        # Some accepted prose denominators are not physical Pint units
        # ("label", "GPU-hour", "device/year"). They can still decorate a
        # pure dollar quantity, but a Pint rate needs a Pint denominator.
        if unit_label == "dollar":
            return amount.m_as(USD)
        raise ValueError(
            f"fmt_usd cannot infer a Pint denominator from per={per!r} for "
            f"quantity units {amount.units}. Use a Pint-parsable per= value "
            "or pass an already-extracted numeric magnitude at the boundary."
        )

    return amount.to(USD / denominator).magnitude


def fmt(
    quantity,
    unit=None,
    precision=1,
    commas=True,
    prefix="",
    suffix="",
    approx=False,
    lower_bound=False,
    upper_bound=False,
    trim_trailing_zeros=False,
):
    """
    Format a Pint Quantity (or plain number) for narrative text.
    Returns a MarkdownStr so Quarto inserts the value verbatim (no escape).

    The prefix and suffix arguments collapse the old MarkdownStr(f"...")
    escape-hatch idiom into a single canonical helper. Common uses:

        fmt(value, precision=0, approx=True)       # "~1,000"
        fmt(value, precision=0, lower_bound=True)  # "> 1,000"
        fmt_percent(rate, precision=1, style="symbol")  # "12.4%"
        fmt_qty(bandwidth, unit=MB/second)         # "2.4 MB/s"
        fmt_multiple(speedup)                      # "3.2×"

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

    # Primary formatting. ``precision=None`` opts into the same auto display
    # policy used by the typed helpers while preserving fmt()'s legacy default.
    auto_precision = precision is None
    display_precision = _resolve_display_precision(val, precision)
    fmt_str = f",.{display_precision}f" if commas else f".{display_precision}f"
    result = f"{val:{fmt_str}}"

    _check_fmt_precision(val, display_precision, result)
    if trim_trailing_zeros or auto_precision:
        result = _trim_auto_decimal_zeros(val, result)

    prefix = _display_prefix(
        prefix,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    decorated = f"{prefix}{result}{suffix}"
    out = MarkdownStr(decorated)
    assert isinstance(out, MarkdownStr), (
        "fmt() must return MarkdownStr — this guard exists so a future refactor "
        "of this module cannot silently break Quarto's _repr_markdown_ detection. "
        "See the project math rules."
    )
    return out


def _round_half_away_from_zero(val):
    """Round to the nearest integer with .5 ties going away from zero.

    Python's built-in ``round()`` uses banker's rounding (2.5 → 2, 3.5 → 4),
    which renders adjacent half-values inconsistently in prose. Decimal
    ``ROUND_HALF_UP`` rounds ties away from zero for both signs
    (2.5 → 3, -2.5 → -3) and works on the decimal repr, so near-integer
    float noise (7.9999999999) still lands on the intended integer (8).
    """
    import decimal

    with decimal.localcontext() as ctx:
        ctx.prec = 60
        return int(
            decimal.Decimal(repr(float(val))).quantize(
                decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP
            )
        )


def fmt_int(
    quantity,
    unit=None,
    commas=True,
    prefix="",
    suffix="",
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """
    Format a value as an integer for narrative text.

    Explicit source-level opt-in for integer display of computed values.
    Rounds half away from zero (2.5 → 3, -2.5 → -3) and delegates to
    ``fmt(..., precision=0, ...)`` so commas and approx markers still apply.

    Safety: a nonzero value that would round to ``0`` raises instead of
    silently rendering as zero (same contract as ``fmt``'s precision guard).
    """
    if unit:
        if isinstance(quantity, ureg.Quantity):
            quantity = quantity.to(unit)
    val = _numeric_magnitude(quantity)
    rounded = _round_half_away_from_zero(val)
    if rounded == 0 and abs(val) > 1e-12:
        raise ValueError(
            f"Formatting Precision Error: Value {val} rounds to '0' under "
            f"fmt_int. This hides the actual value. Use fmt({val!r}, "
            f"precision=...) with enough precision to show the magnitude, "
            f"or change units so integer display is meaningful."
        )
    return fmt(
        rounded,
        precision=0,
        commas=commas,
        prefix=prefix,
        suffix=suffix,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


_USD_MARKERS = {"*"}


def fmt_usd(
    amount,
    *,
    precision=0,
    commas=True,
    approx=False,
    scale=None,
    per=None,
    marker="",
    suffix="",
):
    """
    Canonical currency formatter — the single, blessed way to render any
    dollar amount in prose, tables, or callouts.

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
                per="year")                  # "~\\$1,234/year"
        fmt_usd(gpt3_cost, precision=1,
                scale="M")                   # "\\$4.6M"
        fmt_usd(4_750_000, precision=2,
                scale="million")             # "\\$4.75 million"
        fmt_usd(rate_per_gb, precision=2,
                commas=False, per="GB")      # "\\$0.09/GB"

    Args:
        amount: A plain number, a pure-dollar Pint ``Quantity``, or a Pint
            dollar rate when ``per=`` is a Pint-parsable denominator such as
            ``"GB"``, ``"TB/month"``, ``"hour"``, or ``"kWh"``. Non-physical
            denominators such as ``"label"`` still decorate plain numbers or
            pure-dollar quantities.
        precision: Decimal places. ``precision=0`` rounds to whole dollars
            (``fmt_int`` semantics); ``precision>=1`` uses ``fmt`` semantics
            with its spurious-zero guard.
        commas: Thousands separators (default ``True`` — currency usually
            groups: ``\\$15,000``). Pass ``False`` for small rates.
        approx: When ``True``, prepend ``~`` before the dollar sign.
        scale: Optional currency scale (``"K"``, ``"M"``, ``"B"``, ``"T"`` or
            word forms such as ``"million"``). The raw dollar amount is divided
            by the scale here, so the magnitude and display suffix cannot drift
            apart.
        per: Optional rate denominator (``"month"``, ``"GB"``, ``"kWh"``, or a
            Pint unit such as ``GB``). Pass without a leading slash.
        marker: Optional checked table marker appended after the currency value.
            Currently only ``"*"`` is allowlisted; this is for data-source
            markers, not units or scale glyphs.
        suffix: Legacy escape hatch while the corpus is being migrated. New
            QMD code should use ``scale=`` and/or ``per=`` instead.
    """
    amount = _numeric_magnitude(_usd_amount_magnitude(amount, per))

    # Conventional sign-first rendering: -$500, never $-500. The sign is
    # extracted here so scale division and the escaped \$ prefix below always
    # operate on the absolute amount (works for all scale=/per= variants).
    negative = amount < 0
    if negative:
        amount = -amount

    if marker and marker not in _USD_MARKERS:
        raise ValueError(
            f"fmt_usd marker must be one of {sorted(_USD_MARKERS)}, " f"got {marker!r}."
        )
    if suffix and (scale is not None or per is not None or marker):
        raise ValueError("Use suffix= or structured scale=/per=/marker=, not both.")
    structured_suffix = ""
    if scale is not None:
        divisor, scale_suffix = _resolve_decimal_scale(
            scale,
            what="fmt_usd scale",
        )
        amount = amount / divisor
        structured_suffix += scale_suffix
    structured_suffix += _denominator_suffix(
        per,
        what="fmt_usd per",
        allowed=_USD_DENOMINATORS,
    )
    suffix = suffix or structured_suffix
    suffix += marker

    # approx + negative reads "~-\$500": the approximation marker scopes the
    # whole signed amount.
    prefix = f"{'~' if approx else ''}{'-' if negative else ''}\\$"

    if precision == 0:
        return fmt_int(amount, commas=commas, prefix=prefix, suffix=suffix)
    return fmt(amount, precision=precision, commas=commas, prefix=prefix, suffix=suffix)


def fmt_eur(
    amount,
    *,
    precision=0,
    commas=True,
    scale=None,
    approx=False,
):
    """Format euro-denominated amounts using explicit ``EUR`` prose style."""
    from .core.units import EUR

    if isinstance(amount, ureg.Quantity):
        amount = amount.to(EUR).magnitude
    if scale is not None:
        divisor, scale_suffix = _resolve_decimal_scale(
            scale,
            what="fmt_eur scale",
        )
        amount = amount / divisor
    else:
        scale_suffix = ""
    prefix = "~EUR " if approx else "EUR "
    if precision == 0:
        return fmt_int(amount, commas=commas, prefix=prefix, suffix=scale_suffix)
    return fmt(amount, precision=precision, commas=commas, prefix=prefix, suffix=scale_suffix)


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

    Pairs with ``fmt_val()`` for value/unit table columns. Labels are
    normalized through the shared ``_UNIT_LABEL_NORMALIZATION`` table so a
    unit column always matches the suffix ``fmt_qty`` would render for the
    same quantity (``EFLOPs/second`` → ``EFLOP/s``, ``TFLOPs`` → ``TFLOP``).

    >>> fmt_unit(80 * GB)        # "GB"
    >>> fmt_unit(80)             # "-"
    """
    if isinstance(quantity, ureg.Quantity):
        out = MarkdownStr(_normalize_unit_label(f"{quantity.units}"))
    else:
        out = MarkdownStr(default)
    assert isinstance(out, MarkdownStr), "fmt_unit() must return MarkdownStr"
    return out


def fmt_percent(
    ratio, precision=None, commas=False, style="number", max_ratio=1.5, allow_negative=False
):
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
    and where MIT Press style — spell out "percent" in sentence contexts,
    "%" in dense data contexts — is otherwise unenforceable):

        style="prose"   →  "85 percent"   (MIT Press body style)
        style="symbol"  →  "85%"          (tables, equations, code)
        style="number"  →  "85"           (default; prose carries the meaning,
                                            e.g. "85 MFU", "85 goodput")

    Accepts a Pint Quantity (dimensionless magnitude) or plain float.
    """
    r = _percent_ratio_value(
        ratio,
        max_ratio=max_ratio,
        allow_negative=allow_negative,
        what="fmt_percent",
    )
    if style not in {"number", "prose", "symbol"}:
        raise ValueError(
            f"fmt_percent style must be 'number', 'prose', or 'symbol', " f"got {style!r}."
        )
    glyph = {"number": "", "prose": " percent", "symbol": "%"}[style]
    pct_val = r * 100
    auto_precision = precision is None
    p = _resolve_display_precision(pct_val, precision)
    return fmt(
        pct_val,
        precision=p,
        commas=commas,
        suffix=glyph,
        trim_trailing_zeros=auto_precision,
    )


def _percent_ratio_value(ratio, *, max_ratio=1.5, allow_negative=False, what="fmt_percent"):
    """Coerce and validate the canonical 0-1 percentage ratio domain."""
    if isinstance(ratio, ureg.Quantity):
        r = float(ratio.m_as(""))
    else:
        r = float(ratio)
    _require_finite(r, "percent ratio")

    lo = -max_ratio if allow_negative else 0.0
    if not (lo <= r <= max_ratio):
        raise ValueError(
            f"{what} expects a 0-1 ratio, got {r}. If this is an "
            f"already-scaled 0-100 value, divide by 100 at the source so the "
            f"value flows through {what} as a ratio. If a value above "
            f"{max_ratio * 100:.0f}% is genuinely intended (e.g. growth), pass "
            f"max_ratio= explicitly; if it can be negative (e.g. ROI, a signed "
            f"change), pass allow_negative=True. This guard prevents silent 100x "
            f"errors such as 0.85 -> '8500 percent'."
        )
    return r


_AUTO_INTEGER_REL_TOL = 0.01
_AUTO_MAX_PRECISION = 6


def _formatted_number(val, precision):
    """Format *val* without commas for precision-policy checks."""
    return f"{val:.{precision}f}"


def _relative_error(actual, displayed):
    """Relative error of a displayed scalar; used only for auto display policy."""
    if abs(actual) <= 1e-12:
        return 0.0 if abs(displayed) <= 1e-12 else float("inf")
    return abs(displayed - actual) / abs(actual)


def _resolve_display_precision(val, precision):
    """Pick precision when caller passes precision=None.

    Auto mode should produce the shortest display that still carries the value's
    useful magnitude. In particular, a value such as 153.016 should render as
    ``153`` rather than ``153.0``, while 2.04 should keep two decimals instead
    of collapsing to ``2.0`` or ``2``.
    """
    if precision is not None:
        return precision
    if _is_integer_like(val):
        return 0

    one_decimal = _formatted_number(val, 1)
    one_decimal_value = _parse_formatted_number(one_decimal)
    integer_value = _parse_formatted_number(_formatted_number(val, 0))
    if (
        one_decimal_value is not None
        and one_decimal_value != 0.0
        and (
            not _has_spurious_zero_decimals(one_decimal)
            or (
                integer_value is not None
                and integer_value != 0.0
                and _relative_error(val, integer_value) <= _AUTO_INTEGER_REL_TOL
            )
        )
    ):
        return 1

    for p in range(2, _AUTO_MAX_PRECISION + 1):
        candidate = _formatted_number(val, p)
        candidate_value = _parse_formatted_number(candidate)
        if candidate_value == 0.0:
            continue
        if not _has_spurious_zero_decimals(candidate):
            return p

    return _AUTO_MAX_PRECISION


def _trim_auto_decimal_zeros(val, result):
    """Trim formatter-chosen ``.0`` when integer display is still faithful."""
    if not _has_spurious_zero_decimals(result):
        return result
    integer_display = f"{val:,.0f}" if "," in result else _formatted_number(val, 0)
    integer_value = _parse_formatted_number(integer_display)
    if (
        integer_value is not None
        and integer_value != 0.0
        and _relative_error(val, integer_value) <= _AUTO_INTEGER_REL_TOL
    ):
        return integer_display
    return result


def _range_precisions(precision, *, what="range precision"):
    """Return endpoint precisions for a range helper."""
    if isinstance(precision, (tuple, list)):
        if len(precision) != 2:
            raise ValueError(f"{what} tuple/list must have exactly two entries.")
        lo_p, hi_p = precision
    else:
        lo_p = hi_p = precision
    if lo_p is not None and (not isinstance(lo_p, int) or not isinstance(hi_p, int)):
        raise TypeError(f"{what} must be an int or a 2-item int tuple/list.")
    return lo_p, hi_p


def _range_endpoint_precisions(lo_val, hi_val, precision, *, what="range precision"):
    """Resolve per-endpoint precision; None → auto like fmt_multiple."""
    if isinstance(precision, (tuple, list)):
        return _range_precisions(precision, what=what)
    if precision is not None:
        return precision, precision
    return (
        _resolve_display_precision(lo_val, None),
        _resolve_display_precision(hi_val, None),
    )


def fmt_percent_range(
    lo,
    hi,
    *,
    precision=None,
    commas=False,
    style="prose",
    max_ratio=1.5,
    allow_negative=False,
):
    """Format a range of percent ratios with one checked percent suffix.

    Inputs use the same canonical domain as ``fmt_percent``: ratios in ``[0, 1]``
    by default, not already-scaled 0-100 values. ``precision`` may be one int or
    a ``(lo_precision, hi_precision)`` pair when existing prose intentionally
    uses asymmetric endpoint precision.
    """
    lo_r = _percent_ratio_value(
        lo,
        max_ratio=max_ratio,
        allow_negative=allow_negative,
        what="fmt_percent_range",
    )
    hi_r = _percent_ratio_value(
        hi,
        max_ratio=max_ratio,
        allow_negative=allow_negative,
        what="fmt_percent_range",
    )
    if hi_r < lo_r:
        raise ValueError(f"fmt_percent_range expects hi >= lo, got lo={lo_r}, hi={hi_r}.")
    if style not in {"number", "prose", "symbol"}:
        raise ValueError(
            f"fmt_percent_range style must be 'number', 'prose', or 'symbol', " f"got {style!r}."
        )
    lo_p, hi_p = _range_endpoint_precisions(
        lo_r * 100,
        hi_r * 100,
        precision,
        what="fmt_percent_range precision",
    )
    auto_precision = precision is None
    a = fmt(
        lo_r * 100,
        precision=lo_p,
        commas=commas,
        trim_trailing_zeros=auto_precision,
    )
    b = fmt(
        hi_r * 100,
        precision=hi_p,
        commas=commas,
        trim_trailing_zeros=auto_precision,
    )
    suffix = {"number": "", "prose": " percent", "symbol": "%"}[style]
    return MarkdownStr(f"{a}\u2013{b}{suffix}")


def fmt_pp(points, precision=None, commas=False, style="prose", attributive=False):
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
        v = float(points.m_as(""))
    else:
        v = float(points)
    _require_finite(v, "percentage points")
    if style not in {"prose", "symbol"}:
        raise ValueError(f"fmt_pp style must be 'prose' or 'symbol', got {style!r}.")
    if attributive and style != "prose":
        raise ValueError(
            "fmt_pp(attributive=True) applies only to style='prose' (the "
            "hyphenated adjective form), not style='symbol'."
        )
    auto_precision = precision is None
    num = str(
        fmt(
            v,
            precision=_resolve_display_precision(v, precision),
            commas=commas,
            trim_trailing_zeros=auto_precision,
        )
    )
    if style == "symbol":
        return MarkdownStr(f"{num} pp")
    if attributive:
        return MarkdownStr(f"{num} percentage-point")
    # Singular agrees with the absolute rendered magnitude: "1 percentage
    # point" AND "-1 percentage point" (a string compare on "1" missed the
    # negative case and rendered "-1 percentage points"; fixed 2026-06).
    num_value = _parse_formatted_number(num)
    singular = num_value is not None and abs(num_value) == 1
    word = "percentage point" if singular else "percentage points"
    return MarkdownStr(f"{num} {word}")


def fmt_multiple(
    factor,
    precision=None,
    commas=False,
    *,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """
    Format a multiplier / speedup / scaling factor with its times glyph.

    The multiplier glyph is part of the value-kind, so this helper owns it:

        speedup_mult_str = fmt_multiple(3.2)       # "3.2×"
        # prose: `{python} speedup_mult_str` faster

    Guard: a multiplier is non-negative. Do not use this helper for values
    substituted inside display math; use ``fmt_ratio`` for the bare coefficient
    and keep ``\\times`` inside the equation. The helper emits a literal
    rendered ``×`` glyph because Quarto inline substitutions do not reliably
    re-parse math delimiters inside returned strings.
    """
    v = _dimensionless_magnitude(factor, what="multiplier")
    if v < 0:
        raise ValueError(
            f"fmt_multiple expects a non-negative factor, got {v}. A negative "
            f"multiplier almost always signals a sign or ordering bug."
        )
    auto_precision = precision is None
    p = _resolve_display_precision(v, precision)
    num = fmt(v, precision=p, commas=commas, trim_trailing_zeros=auto_precision)
    prefix = _display_prefix(approx=approx, lower_bound=lower_bound, upper_bound=upper_bound)
    return MarkdownStr(f"{prefix}{num}×")


def fmt_multiple_range(lo, hi, *, precision=1, commas=False):
    """Format a range of multiplier factors with one trailing times glyph.

    ``precision`` may be one int or a ``(lo_precision, hi_precision)`` pair.
    """
    lo_v = _dimensionless_magnitude(lo, what="multiplier range endpoint")
    hi_v = _dimensionless_magnitude(hi, what="multiplier range endpoint")
    if lo_v < 0 or hi_v < 0:
        raise ValueError(
            f"fmt_multiple_range expects non-negative factors, got lo={lo_v}, " f"hi={hi_v}."
        )
    if hi_v < lo_v:
        raise ValueError(f"fmt_multiple_range expects hi >= lo, got lo={lo_v}, hi={hi_v}.")
    lo_p, hi_p = _range_endpoint_precisions(
        lo_v,
        hi_v,
        precision,
        what="fmt_multiple_range precision",
    )
    auto_precision = precision is None
    a = fmt(lo_v, precision=lo_p, commas=commas, trim_trailing_zeros=auto_precision)
    b = fmt(hi_v, precision=hi_p, commas=commas, trim_trailing_zeros=auto_precision)
    return MarkdownStr(f"{a}\u2013{b}×")


def fmt_count(
    value,
    scale=None,
    precision=0,
    commas=True,
    label=None,
    plural_label=None,
    suffix="",
    approx=False,
    lower_bound=False,
    allow_fractional=False,
    scale_style="symbol",
    attributive=False,
):
    """
    Format a **count** (a dimensionless tally of things), optionally with a
    magnitude scale glyph.

    Replaces the ``fmt(value / MILLION, suffix="M")`` idiom, where the scale
    division and the glyph were two disconnected facts a reader had to keep in
    sync. Here the scale is declared once and applied for you:

        fmt_count(5_300_000, scale="M")              # "5M"
        fmt_count(5_300_000, scale="M", precision=1) # "5.3M"
        fmt_count(5_300_000, scale="M",
                  scale_style="word", precision=1)   # "5.3 million"
        fmt_count(5_300_000, scale="million",
                  precision=1)                       # "5.3 million"
        fmt_count(70e9, scale="B")                   # "70B"   (e.g. params)
        fmt_count(7e9, scale="billion",
                  attributive=True)                  # "7-billion"
        fmt_count(8192)                              # "8,192" (no scale)
        fmt_count(1024, label="GPU")                 # "1,024 GPUs"
        fmt_count(1024, label="GPU", approx=True)    # "~1,024 GPUs"
        fmt_count(2, label="batch",
                  plural_label="batches")            # "2 batches"

    Guard: counts are non-negative. When a count noun ``label`` is provided,
    the raw value must be whole-number by default; pass
    ``allow_fractional=True`` only when a fractional labeled count is
    intentionally wanted. Label strings are count nouns; units/rates/glyphs
    must use their own typed formatter.

    For a currency amount use ``fmt_usd``; for a physical quantity use
    ``fmt_qty``. ``fmt_count`` is for pure tallies only.
    """
    raw_v = _numeric_magnitude(value)
    require_integer = label is not None or plural_label is not None or attributive
    _validate_count_value(
        raw_v,
        allow_fractional=allow_fractional,
        require_integer=require_integer,
    )
    if suffix and (label is not None or plural_label is not None):
        raise ValueError("Use suffix= or structured label=, not both.")
    if attributive and (suffix or label is not None or plural_label is not None):
        raise ValueError(
            "fmt_count attributive=True formats only the scaled modifier; "
            "put the count noun in prose."
        )
    if scale_style not in {"symbol", "word"}:
        raise ValueError(
            "fmt_count scale_style must be 'symbol' or 'word', " f"got {scale_style!r}."
        )
    if scale is None and scale_style != "symbol":
        raise ValueError("fmt_count scale_style='word' requires scale=.")
    if scale is None and attributive:
        raise ValueError("fmt_count attributive=True requires scale=.")
    v = raw_v
    glyph = ""
    if scale is not None:
        divisor, glyph = _resolve_decimal_scale(
            scale,
            what="fmt_count scale",
            style=scale_style,
            attributive=attributive,
        )
        v = v / divisor
    suffix = suffix or _label_suffix(raw_v, label, plural_label)
    return fmt(
        v,
        precision=precision,
        commas=commas,
        suffix=glyph + suffix,
        approx=approx,
        lower_bound=lower_bound,
    )


def _fmt_scaled_count_quantity(
    value,
    *,
    unit,
    scale,
    precision,
    commas,
    what,
    scale_style="symbol",
    attributive=False,
    approx=False,
    lower_bound=False,
):
    raw_v = _numeric_magnitude(value.to(unit) if isinstance(value, ureg.Quantity) else value)
    _validate_count_value(raw_v, allow_fractional=False, require_integer=True)
    auto_scaled = scale == "auto"
    scale = _auto_decimal_scale(raw_v) if auto_scaled else scale
    divisor, glyph = _resolve_decimal_scale(
        scale,
        what=f"{what} scale",
        style=scale_style,
        attributive=attributive,
    )
    v = raw_v / divisor
    p = _resolve_scaled_count_precision(v, precision)
    if auto_scaled and scale is not None:
        # Boundary fix (2026-06): _auto_decimal_scale picks the divisor from
        # the RAW value, so 999,999,999 lands on "M" and then ROUNDS to a
        # 1000.0M mantissa at display precision. When the rounded mantissa
        # reaches 1000, promote to the next larger scale (M→B→T) so the
        # display reads "1B", not "1000.0M". At the largest scale (T) there
        # is nowhere to promote; keep the value and group digits with commas.
        order = sorted(_DECIMAL_SCALE_FACTORS, key=_DECIMAL_SCALE_FACTORS.get)
        while abs(float(f"{v:.{p}f}")) >= 1000:
            idx = order.index(scale)
            if idx + 1 >= len(order):
                commas = True
                break
            scale = order[idx + 1]
            divisor, glyph = _resolve_decimal_scale(
                scale,
                what=f"{what} scale",
                style=scale_style,
                attributive=attributive,
            )
            v = raw_v / divisor
            p = _resolve_scaled_count_precision(v, precision)
    return fmt(
        v,
        precision=p,
        commas=commas,
        suffix=glyph,
        approx=approx,
        lower_bound=lower_bound,
    )


def fmt_params(
    parameters,
    *,
    scale="auto",
    precision=None,
    commas=False,
    scale_style="symbol",
    attributive=False,
    approx=False,
    lower_bound=False,
):
    """Format model-parameter counts as checked count quantities.

    Accepts a Pint ``param``/``count`` quantity or a raw parameter count. The
    default auto-scales to compact model-size strings such as ``175B`` and
    ``150M`` while preserving decimals when needed, such as ``1.5B``. Pass an
    explicit scale when a comparison needs a fixed denominator, e.g.
    ``scale="B"`` to render ``150M`` as ``0.15B``.
    """
    from .core.units import param

    return _fmt_scaled_count_quantity(
        parameters,
        unit=param,
        scale=scale,
        precision=precision,
        commas=commas,
        what="fmt_params",
        scale_style=scale_style,
        attributive=attributive,
        approx=approx,
        lower_bound=lower_bound,
    )


def fmt_tokens(
    tokens,
    *,
    scale=None,
    precision=None,
    commas=True,
    scale_style="symbol",
    attributive=False,
    approx=False,
    lower_bound=False,
):
    """Format token counts as checked count quantities."""
    from .core.units import count

    return _fmt_scaled_count_quantity(
        tokens,
        unit=count,
        scale=scale,
        precision=precision,
        commas=commas,
        what="fmt_tokens",
        scale_style=scale_style,
        attributive=attributive,
        approx=approx,
        lower_bound=lower_bound,
    )


_RATE_UNITS = {
    "QPS",
    "FPS",
    "IOPS",
    "MIPS",
    "MACs/cycle",
    "tokens/s",
    "tokens/hour",
    "img/s",
    "images/s",
    "queries/s",
    "req/s",
    "requests/s",
    "requests/day",
    "events/s",
    "errors/hour",
    "failures/hour",
    "failures/day",
    "inferences/s",
    "inferences/hour",
    "inferences/day",
    "patients/day",
    "queries/hour",
    "queries/day",
    "reviews/day",
    "samples/s",
    "samples/min",
    "samples/hour",
    "deploys/week",
    "checkpoints/hour",
    "windows/month",
    "false wake/month",  # singular: a rate of exactly 1 must not read "1 false wakes"
    "false wakes/month",
    "feature reads/request",
    "queries/month",
    "queries/class",
    "incidents/year",
    "models/year",
    "reads/year",
    "photos/patient",
    "utterances/speaker",
    "cameras/store",
    "boards/store",
    "cases/day",
    "km/h",
}


def fmt_rate(
    value,
    unit,
    *,
    precision=0,
    commas=True,
    scale=None,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    allow_negative=False,
):
    """Format a non-physical count throughput such as QPS or tokens/s.

    Physical rates (``GB/s``, ``TFLOP/s``, ``W``) remain ``fmt_qty`` values.
    This helper is for named service/data rates whose numerator is a counted
    event rather than a Pint physical unit. ``scale=`` accepts the same checked
    decimal scales as ``fmt_count``.
    """
    unit = _clean_text_atom(unit, what="rate unit")
    if unit not in _RATE_UNITS:
        raise ValueError(f"fmt_rate unit must be one of {sorted(_RATE_UNITS)}, got {unit!r}.")
    v = _numeric_magnitude(value)
    if v < 0 and not allow_negative:
        raise ValueError(
            f"fmt_rate expects a non-negative rate, got {v}. Pass "
            f"allow_negative=True if a signed rate is genuinely intended."
        )
    scale_suffix = ""
    if scale is not None:
        divisor, scale_suffix = _resolve_decimal_scale(
            scale,
            what="fmt_rate scale",
        )
        v = v / divisor
    return fmt(
        v,
        precision=precision,
        commas=commas,
        suffix=f"{scale_suffix} {unit}",
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def fmt_fps(
    value,
    *,
    precision=0,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    allow_negative=False,
):
    """Format frames per second for vision/video workloads."""
    v = _numeric_magnitude(value)
    if v < 0 and not allow_negative:
        raise ValueError(
            f"fmt_fps expects a non-negative rate, got {v}. Pass "
            f"allow_negative=True if a signed frame-rate delta is genuinely intended."
        )
    if precision == 0:
        return fmt_int(
            v,
            commas=commas,
            suffix=" FPS",
            approx=approx,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    return fmt_rate(
        v,
        "FPS",
        precision=precision,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        allow_negative=allow_negative,
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
    v = _dimensionless_magnitude(value, what="ratio")
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


def fmt_qty_range(
    lo,
    hi,
    display_unit=None,
    *,
    unit=None,
    precision=None,
    commas=False,
    unit_label=None,
    per=None,
):
    """Format a range of Pint quantities with one canonical unit suffix."""
    if not isinstance(lo, ureg.Quantity) or not isinstance(hi, ureg.Quantity):
        raise TypeError(
            "fmt_qty_range() requires Pint Quantity endpoints. Keep units "
            "attached until formatting."
        )
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_qty_range")
    lo_v = _numeric_magnitude(lo.to(display_unit))
    hi_v = _numeric_magnitude(hi.to(display_unit))
    if hi_v < lo_v:
        raise ValueError(f"fmt_qty_range expects hi >= lo, got lo={lo_v}, hi={hi_v}.")
    lo_p, hi_p = _range_endpoint_precisions(lo_v, hi_v, precision, what="fmt_qty_range precision")
    auto_precision = precision is None
    a = fmt(lo_v, precision=lo_p, commas=commas, trim_trailing_zeros=auto_precision)
    b = fmt(hi_v, precision=hi_p, commas=commas, trim_trailing_zeros=auto_precision)
    suffix = _quantity_suffix(display_unit, unit_label=unit_label, per=per)
    return MarkdownStr(f"{a}\u2013{b}{suffix}")


def fmt_sci_qty(
    quantity,
    display_unit=None,
    *,
    unit=None,
    precision=2,
    unit_label=None,
    per=None,
):
    """Format a Pint quantity in scientific notation with a checked unit suffix."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError(
            "fmt_sci_qty() requires a Pint Quantity. Keep units attached at "
            "the call site, e.g. fmt_sci_qty(ops, flop), not "
            "fmt_sci_qty(ops.magnitude, flop)."
        )
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_sci_qty")
    q = quantity.to(display_unit)
    body = fmt_sci(q.magnitude, precision=precision)
    suffix = _quantity_suffix(display_unit, unit_label=unit_label, per=per)
    return MarkdownStr(f"{body}{suffix}")


def fmt_sci_flops(quantity, *, unit=None, precision=2):
    """Format FLOP work/count quantities in scientific notation.

    QMD code should not spell ``unit_label="FLOPs"`` at each call site. This
    helper keeps the Pint conversion checked while centralizing the prose
    convention that scalar floating-point operation counts are rendered as
    plural ``FLOPs`` in scientific notation.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_sci_flops() requires a Pint Quantity.")
    from .core.units import flop

    display_unit = _coerce_unit(unit) if unit is not None else flop
    return fmt_sci_qty(
        quantity,
        display_unit,
        precision=precision,
        unit_label="FLOPs",
    )


def fmt_time_range(
    lo,
    hi,
    display_unit=None,
    *,
    unit=None,
    precision=None,
    commas=False,
    style="symbol",
):
    """Format a duration range with time-unit validation."""
    display_unit = _resolve_unit_arg(display_unit, unit, what="fmt_time_range")
    one = 1 * display_unit
    if not one.check("[time]"):
        raise ValueError(f"fmt_time_range display_unit must be a time unit, got {display_unit}.")
    lo_q = lo if isinstance(lo, ureg.Quantity) else lo * display_unit
    hi_q = hi if isinstance(hi, ureg.Quantity) else hi * display_unit
    if style == "symbol":
        return fmt_qty_range(
            lo_q,
            hi_q,
            display_unit,
            precision=precision,
            commas=commas,
        )
    if style != "word":
        raise ValueError(f"fmt_time_range style must be 'symbol' or 'word', got {style!r}.")
    lo_v = _numeric_magnitude(lo_q.to(display_unit))
    hi_v = _numeric_magnitude(hi_q.to(display_unit))
    if hi_v < lo_v:
        raise ValueError(f"fmt_time_range expects hi >= lo, got lo={lo_v}, hi={hi_v}.")
    lo_p, hi_p = _range_endpoint_precisions(lo_v, hi_v, precision, what="fmt_time_range precision")
    auto_precision = precision is None
    a = fmt(lo_v, precision=lo_p, commas=commas, trim_trailing_zeros=auto_precision)
    b = fmt(hi_v, precision=hi_p, commas=commas, trim_trailing_zeros=auto_precision)
    label_value = 1 if abs(lo_v - 1) <= 1e-9 and abs(hi_v - 1) <= 1e-9 else 2
    suffix = _time_word_suffix(label_value, display_unit)
    return MarkdownStr(f"{a}\u2013{b}{suffix}")


def fmt_count_range(
    lo,
    hi,
    *,
    scale=None,
    precision=0,
    commas=True,
    label=None,
    plural_label=None,
    allow_fractional=False,
):
    """Format a range of counts with optional scale and count noun."""
    lo_raw = _numeric_magnitude(lo)
    hi_raw = _numeric_magnitude(hi)
    require_integer = label is not None or plural_label is not None
    _validate_count_value(
        lo_raw,
        allow_fractional=allow_fractional,
        require_integer=require_integer,
    )
    _validate_count_value(
        hi_raw,
        allow_fractional=allow_fractional,
        require_integer=require_integer,
    )
    if hi_raw < lo_raw:
        raise ValueError(f"fmt_count_range expects hi >= lo, got lo={lo_raw}, hi={hi_raw}.")
    lo_count = lo_raw
    hi_count = hi_raw
    glyph = ""
    if scale is not None:
        divisor, glyph = _resolve_decimal_scale(
            scale,
            what="fmt_count_range scale",
        )
        lo_raw = lo_raw / divisor
        hi_raw = hi_raw / divisor
    a = fmt(lo_raw, precision=precision, commas=commas)
    b = fmt(hi_raw, precision=precision, commas=commas)
    suffix = ""
    if label is not None or plural_label is not None:
        label_value = 1 if abs(lo_count - 1) <= 1e-9 and abs(hi_count - 1) <= 1e-9 else 2
        suffix = _label_suffix(label_value, label, plural_label)
    return MarkdownStr(f"{a}{glyph}\u2013{b}{glyph}{suffix}")


def fmt_usd_range(
    lo,
    hi,
    *,
    precision=0,
    commas=True,
    scale=None,
    per=None,
    approx=False,
    repeat_symbol=True,
):
    """Format a currency range with optional scale and one denominator."""
    from .core.units import USD

    if isinstance(lo, ureg.Quantity):
        lo = lo.m_as(USD)
    if isinstance(hi, ureg.Quantity):
        hi = hi.m_as(USD)
    lo_v = _numeric_magnitude(lo)
    hi_v = _numeric_magnitude(hi)
    if hi_v < lo_v:
        raise ValueError(f"fmt_usd_range expects hi >= lo, got lo={lo_v}, hi={hi_v}.")
    a = fmt_usd(
        lo_v,
        precision=precision,
        commas=commas,
        scale=scale,
        approx=approx,
    )
    if repeat_symbol:
        b = fmt_usd(hi_v, precision=precision, commas=commas, scale=scale)
    else:
        hi_display = hi_v
        suffix = ""
        if scale is not None:
            divisor, scale_suffix = _resolve_decimal_scale(
                scale,
                what="fmt_usd_range scale",
            )
            hi_display = hi_display / divisor
            suffix = scale_suffix
        if precision == 0:
            b = fmt_int(hi_display, commas=commas, suffix=suffix)
        else:
            b = fmt(
                hi_display,
                precision=precision,
                commas=commas,
                suffix=suffix,
            )
    suffix = _denominator_suffix(
        per,
        what="fmt_usd_range per",
        allowed=_USD_DENOMINATORS,
    )
    return MarkdownStr(f"{a}\u2013{b}{suffix}")


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
        label = f"{_coerce_unit(display_unit):~P}"
    except Exception:
        label = f"{display_unit}"
    return f" {_normalize_unit_label(label)}"


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


def _pick_length_unit(qty):
    """Auto-select meter vs kilometer for prose length display."""
    from .core.units import kilometer, meter

    q = qty.to(meter)
    mag = abs(q.magnitude)
    if mag < 1000:
        return meter
    return kilometer


def _pick_area_unit(qty):
    """Auto-select square millimeters for chip-scale areas, else square meters."""
    from .core.units import meter

    q = qty.to(meter**2)
    if abs(q.magnitude) < 0.01:
        return ureg.millimeter**2
    return meter**2


def _pick_memory_unit(qty, *, binary=False):
    from .core.units import GB, GiB, KB, KiB, MB, MiB, PB, TB, TiB, byte

    q = qty.to(byte)
    mag = abs(q.magnitude)
    if binary:
        if mag < 1024**2:
            return KiB
        if mag < 1024**3:
            return MiB
        if mag < 1024**4:
            return GiB
        return TiB
    if mag < 1e6:
        return KB
    if mag < 1e9:
        return MB
    if mag < 1e12:
        return GB
    if mag < 1e15:
        return TB
    return PB


def _pick_bandwidth_unit(qty):
    from .core.units import GB, MB, TB, byte, second

    q = qty.to(byte / second)
    mag = abs(q.magnitude)
    if mag < 1e9:
        return MB / second
    if mag < 1e12:
        return GB / second
    return TB / second


def _pick_flop_rate_unit(qty):
    from .core.units import EFLOP, GFLOP, PFLOP, TFLOP, ZFLOP, second

    q = qty.to(GFLOP / second)
    mag = abs(q.magnitude)
    if mag < 1e3:
        return GFLOP / second
    tflops = q.to(TFLOP / second).magnitude
    if abs(tflops) < 1e3:
        return TFLOP / second
    pflops = q.to(PFLOP / second).magnitude
    if abs(pflops) < 1e3:
        return PFLOP / second
    eflops = q.to(EFLOP / second).magnitude
    if abs(eflops) < 1e3:
        return EFLOP / second
    return ZFLOP / second


def _pick_flops_unit(qty):
    from .core.units import EFLOP, GFLOP, KFLOP, MFLOP, PFLOP, TFLOP, ZFLOP, flop

    q = qty.to(flop)
    mag = abs(q.magnitude)
    if mag < 1e6:
        return KFLOP
    if mag < 1e9:
        return MFLOP
    if mag < 1e12:
        return GFLOP
    if mag < 1e15:
        return TFLOP
    if mag < 1e18:
        return PFLOP
    if mag < 1e21:
        return EFLOP
    return ZFLOP


def _pick_ops_rate_unit(qty):
    from .core.units import GOPS, KOPS, MOPS, OPS, TOPS

    q = qty.to(OPS)
    mag = abs(q.magnitude)
    if mag < 1e3:
        return OPS
    if mag < 1e6:
        return KOPS
    if mag < 1e9:
        return MOPS
    if mag < 1e12:
        return GOPS
    return TOPS


def _pick_power_unit(qty):
    from .core.units import GW, MW, kilowatt, watt

    q = qty.to(watt)
    mag = abs(q.magnitude)
    if mag >= 1e9:
        return GW
    if mag >= 1e6:
        return MW
    if mag >= 1e3:
        return kilowatt
    return watt


def _pick_energy_unit(qty):
    from .core.units import GWh, MWh, Wh, joule, kWh

    q = qty.to(joule)
    mag = abs(q.magnitude)
    if mag < 3600:
        return joule
    wh = q.to(Wh).magnitude
    if wh < 1e3:
        return Wh
    kwh = q.to(kWh).magnitude
    if kwh < 1e3:
        return kWh
    mwh = q.to(MWh).magnitude
    if mwh < 1e3:
        return MWh
    return GWh


def _pick_emissions_unit(qty):
    from .core.units import gram, kilogram, metric_ton

    q = qty.to(gram)
    mag = abs(q.magnitude)
    if mag < 1e3:
        return gram
    kg = q.to(kilogram).magnitude
    if kg < 1e3:
        return kilogram
    return metric_ton


def _pick_carbon_intensity_unit(qty):
    from .core.units import gram, kWh

    # Grid intensities are normally communicated as g/kWh. Callers
    # can force kg/kWh when prose needs that convention.
    qty.to(gram / kWh)
    return gram / kWh


def _pick_latency_unit(qty):
    from .core.units import hour, microsecond, millisecond, minute, nanosecond, second

    q = qty.to(second)
    mag = abs(q.magnitude)
    if mag < 1e-6:
        return nanosecond
    if mag < 1e-3:
        return microsecond
    if mag < 1:
        return millisecond
    if mag < 60:
        return second
    if mag < 3600:
        return minute
    return hour


def fmt_power(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale power quantities for prose (W, kW, MW, GW)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_power() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_power_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_energy(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale energy quantities for prose (J, Wh, kWh, MWh, GWh)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_energy() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_energy_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_bandwidth(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale bandwidth for prose (MB/s, GB/s, TB/s)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_bandwidth() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_bandwidth_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_flop_rate(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale FLOP throughput for prose (GFLOP/s through ZFLOP/s)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_flop_rate() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_flop_rate_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_flops(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    per=None,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale FLOP work/count quantities for prose (KFLOP through ZFLOP)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_flops() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_flops_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        per=per,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_arithmetic_intensity(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format arithmetic intensity quantities as FLOP/byte."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_arithmetic_intensity() requires a Pint Quantity.")
    from .core.units import byte, flop

    display_unit = _coerce_unit(unit) if unit is not None else flop / byte
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_ops_rate(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale non-FLOP operation rates for prose (OPS through TOPS)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_ops_rate() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_ops_rate_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_decibel(quantity, *, unit=None, precision=None, commas=False):
    """Format logarithmic level quantities as dB."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_decibel() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.decibel
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="dB",
        trim_trailing_zeros=auto_precision,
    )


def fmt_illuminance(quantity, *, unit=None, precision=None, commas=False):
    """Format illuminance quantities with the prose label ``lux``."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_illuminance() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.lux
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="lux",
        trim_trailing_zeros=auto_precision,
    )


def fmt_temperature(quantity, *, unit=None, precision=None, commas=False):
    """Format absolute temperatures, defaulting to degrees Celsius."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_temperature() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.degC
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="°C",
        trim_trailing_zeros=auto_precision,
    )


def fmt_temperature_rate(quantity, *, unit=None, precision=None, commas=False):
    """Format temperature-change rates, defaulting to degrees Celsius per second."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_temperature_rate() requires a Pint Quantity.")
    from .core.units import second

    display_unit = _coerce_unit(unit) if unit is not None else ureg.delta_degC / second
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="°C/s",
        trim_trailing_zeros=auto_precision,
    )


def fmt_compute_efficiency(quantity, *, unit=None, precision=None, commas=False):
    """Format accelerator efficiency as FLOP throughput per watt."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_compute_efficiency() requires a Pint Quantity.")
    from .core.units import TFLOP, second, watt

    display_unit = _coerce_unit(unit) if unit is not None else TFLOP / second / watt
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_energy_per_byte(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format data-movement energy intensity, conventionally as pJ/byte."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_energy_per_byte() requires a Pint Quantity.")
    from .core.units import byte, pJ

    display_unit = _coerce_unit(unit) if unit is not None else pJ / byte
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_energy_per_bit(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format signaling or link energy intensity, conventionally as pJ/bit."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_energy_per_bit() requires a Pint Quantity.")
    from .core.units import bit, pJ

    display_unit = _coerce_unit(unit) if unit is not None else pJ / bit
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_energy_per_flop(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format compute energy intensity, conventionally as pJ/FLOP."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_energy_per_flop() requires a Pint Quantity.")
    from .core.units import flop, pJ

    display_unit = _coerce_unit(unit) if unit is not None else pJ / flop
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_energy_per_op(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format generic operation energy, conventionally as pJ/op.

    The registry stores operation energy on the custom ``flop`` dimension so
    compute work remains dimensionally checked. Use this helper when prose is
    discussing a generic arithmetic operation or MAC rather than explicitly
    naming FLOPs.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_energy_per_op() requires a Pint Quantity.")
    from .core.units import flop, pJ

    default_unit = pJ / flop
    display_unit = _coerce_unit(unit) if unit is not None else default_unit
    if (1 * display_unit).dimensionality != (1 * default_unit).dimensionality:
        raise ValueError(f"fmt_energy_per_op unit must be energy/operation, got {display_unit}.")
    q = quantity.to(display_unit)
    if q.magnitude < 0:
        raise ValueError(f"fmt_energy_per_op expects a non-negative energy, got {q.magnitude}.")
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = None
    try:
        if abs((1 * display_unit).to(default_unit).magnitude - 1) < 1e-12:
            unit_label = "pJ/op"
    except Exception:
        unit_label = None
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit_label=unit_label,
        trim_trailing_zeros=auto_precision,
    )


def fmt_length(quantity, *, unit=None, precision=None, commas=None):
    """Format length/distance quantities for prose (m or km).

    When ``unit`` is omitted, values under 1,000 m render in meters; longer
    distances auto-scale to km. Default comma policy: none for meter magnitudes
    below 1,000; commas enabled for km and for meter values ≥ 1,000.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_length() requires a Pint Quantity.")
    from .core.units import meter

    display_unit = _coerce_unit(unit) if unit is not None else _pick_length_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    if commas is None:
        commas = display_unit != meter or abs(q.magnitude) >= 1000
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_area(quantity, *, unit=None, precision=None, commas=False):
    """Format physical areas, defaulting to chip-scale mm² when appropriate."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_area() requires a Pint Quantity.")
    from .core.units import meter

    display_unit = _coerce_unit(unit) if unit is not None else _pick_area_unit(quantity)
    if (1 * display_unit).dimensionality != (1 * meter**2).dimensionality:
        raise ValueError(f"fmt_area unit must be an area unit, got {display_unit}.")
    q = quantity.to(display_unit)
    if q.magnitude < 0:
        raise ValueError(f"fmt_area expects a non-negative area, got {q.magnitude}.")
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_heat_flux(quantity, *, unit=None, precision=None, commas=False):
    """Format heat flux / power density quantities, conventionally as W/cm²."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_heat_flux() requires a Pint Quantity.")
    from .core.units import watt

    default_unit = watt / (ureg.centimeter**2)
    display_unit = _coerce_unit(unit) if unit is not None else default_unit
    if (1 * display_unit).dimensionality != (1 * default_unit).dimensionality:
        raise ValueError(f"fmt_heat_flux unit must be power/area, got {display_unit}.")
    q = quantity.to(display_unit)
    if q.magnitude < 0:
        raise ValueError(f"fmt_heat_flux expects a non-negative heat flux, got {q.magnitude}.")
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_specific_heat(quantity, *, unit=None, precision=None, commas=False):
    """Format specific heat capacity quantities with the textbook J/kg/K label."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_specific_heat() requires a Pint Quantity.")
    from .core.units import J, K, kg

    default_unit = J / (kg * K)
    display_unit = _coerce_unit(unit) if unit is not None else default_unit
    if quantity.dimensionality != (1 * default_unit).dimensionality:
        raise ValueError(f"fmt_specific_heat unit must be energy/mass/temperature, got {quantity.units}.")
    if (1 * display_unit).dimensionality != (1 * default_unit).dimensionality:
        raise ValueError(f"fmt_specific_heat unit must be energy/mass/temperature, got {display_unit}.")
    q = quantity.to(display_unit)
    if q.magnitude <= 0:
        raise ValueError(f"fmt_specific_heat expects a positive specific heat, got {q.magnitude}.")
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="J/kg/K",
        trim_trailing_zeros=auto_precision,
    )


def fmt_memory(quantity, *, unit=None, precision=None, commas=False, binary=False):
    """Auto-scale memory sizes for prose."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_memory() requires a Pint Quantity.")
    display_unit = (
        _coerce_unit(unit) if unit is not None else _pick_memory_unit(quantity, binary=binary)
    )
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def _memory_capacity_unit_label(display_unit):
    """Return vendor-style capacity labels for binary memory display units."""
    from .core.units import GiB, KiB, MiB, TiB

    for binary_unit, label in (
        (KiB, "KB"),
        (MiB, "MB"),
        (GiB, "GB"),
        (TiB, "TB"),
    ):
        try:
            if abs((1 * display_unit).to(binary_unit).magnitude - 1) < 1e-12:
                return label
        except Exception:
            continue
    return None


def fmt_memory_capacity(
    quantity, *, unit=None, precision=None, commas=False, approx=False
):
    """Format branded hardware memory capacities.

    Hardware spec sheets commonly label binary memory capacities with decimal
    symbols such as ``GB``. The registry keeps the Pint quantity intact (often
    ``GiB`` for accelerator memory), while this formatter centralizes the vendor
    display convention: preserve the binary magnitude and render the published
    capacity label. Use ``fmt_memory`` when physical decimal/binary conversion
    should be shown literally.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_memory_capacity() requires a Pint Quantity.")
    display_unit = (
        _coerce_unit(unit) if unit is not None else _pick_memory_unit(quantity, binary=True)
    )
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        unit_label=_memory_capacity_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_emissions(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
):
    """Auto-scale carbon mass for prose (g, kg, metric tons)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_emissions() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_emissions_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        approx=approx,
        lower_bound=lower_bound,
        trim_trailing_zeros=auto_precision,
    )


def fmt_carbon_intensity(quantity, *, unit=None, precision=None, commas=False):
    """Format grid carbon intensity quantities (typically g/kWh)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_carbon_intensity() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_carbon_intensity_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def _water_unit_label(display_unit):
    from .core.units import L, day, hour, kWh

    known = (
        (L, "L"),
        (L / hour, "L/h"),
        (L / day, "L/day"),
        (L / kWh, "L/kWh"),
    )
    for unit, label in known:
        try:
            if abs((1 * display_unit).to(unit).magnitude - 1) < 1e-12:
                return label
        except Exception:
            continue
    return None


def fmt_water(quantity, *, unit=None, precision=None, commas=True):
    """Format water volume quantities with compact liter labels."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_water() requires a Pint Quantity.")
    from .core.units import L

    display_unit = _coerce_unit(unit) if unit is not None else L
    if (1 * display_unit).dimensionality != (1 * L).dimensionality:
        raise ValueError(f"fmt_water unit must be a volume unit, got {display_unit}.")
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label=_water_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_water_rate(quantity, *, unit=None, precision=None, commas=True):
    """Format water flow rates such as L/h and L/day."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_water_rate() requires a Pint Quantity.")
    from .core.units import L, day

    display_unit = _coerce_unit(unit) if unit is not None else L / day
    if (1 * display_unit).dimensionality != (1 * L / day).dimensionality:
        raise ValueError(f"fmt_water_rate unit must be volume/time, got {display_unit}.")
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label=_water_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_water_intensity(quantity, *, unit=None, precision=None, commas=False):
    """Format water usage effectiveness style quantities such as L/kWh."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_water_intensity() requires a Pint Quantity.")
    from .core.units import L, kWh

    display_unit = _coerce_unit(unit) if unit is not None else L / kWh
    if (1 * display_unit).dimensionality != (1 * L / kWh).dimensionality:
        raise ValueError(f"fmt_water_intensity unit must be volume/energy, got {display_unit}.")
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label=_water_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_latency(duration, *, unit=None, precision=None, commas=False):
    """Auto-scale latency durations for prose (ns … hour)."""
    if not isinstance(duration, ureg.Quantity):
        raise TypeError("fmt_latency() requires a Pint Quantity.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_latency_unit(duration)
    q = duration.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_time(
        q,
        display_unit,
        precision=p,
        commas=commas,
        style="symbol",
        trim_trailing_zeros=auto_precision,
    )


def assert_qty_close(actual, expected, display_unit, *, rel=1e-9, abs_tol=None, message=""):
    """Guard helper: compare quantities in display_unit within tolerance."""
    if not isinstance(actual, ureg.Quantity):
        raise TypeError("assert_qty_close actual must be a Pint Quantity.")
    display_unit = _coerce_unit(display_unit)
    actual_mag = actual.to(display_unit).magnitude
    if isinstance(expected, ureg.Quantity):
        expected_mag = expected.to(display_unit).magnitude
    else:
        expected_mag = _numeric_magnitude(expected)
    tol = abs_tol if abs_tol is not None else rel * max(abs(expected_mag), 1.0)
    detail = message or (f"expected {expected_mag} {display_unit}, got {actual_mag} {display_unit}")
    check(abs(actual_mag - expected_mag) <= tol, detail)


def check(condition, message):
    """
    Invariant guard for narrative logic.
    Ensures that calculated values support the surrounding narrative claims.
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


def fmt_display_math(expression):
    """
    Wrap a LaTeX math expression in `$$...$$` for display math output.
    Returns a MarkdownStr.
    """
    out = MarkdownStr(f"$${expression}$$")
    assert isinstance(out, MarkdownStr), "fmt_display_math() must return MarkdownStr"
    return out


def fmt_text(*parts):
    """
    Compose already formatted Markdown fragments and literal text.

    This is the formatter-owned replacement for LEGO-cell
    ``MarkdownStr(f"...{fmt(...)}...")`` assembly. It preserves Quarto's raw
    Markdown rendering hook while keeping scalar formatting in typed helpers.
    """
    out = MarkdownStr("".join(str(part) for part in parts))
    assert isinstance(out, MarkdownStr), "fmt_text() must return MarkdownStr"
    return out
