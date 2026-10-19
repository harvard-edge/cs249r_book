"""Core scalar, currency, percent, and precision-checked formatting primitives."""

from ..core.units import ureg



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

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )


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
    from ..core.provenance import Sourced, scalar_value

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
    if isinstance(display_unit, ureg.Quantity):
        return display_unit.units
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
    from ..core.units import USD

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

    import sys as _sys
    _fmt_pkg = _sys.modules.get("mlsysim.fmt")
    getattr(_fmt_pkg, "_check_fmt_precision", _check_fmt_precision)(val, display_precision, result)
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
    as the escaped ``\\$``. Authors never type a dollar sign and never type ``prefix=``; they call
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
    from ..core.units import EUR

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


def _compact_unit_suffix(display_unit) -> str:
    """Derive a leading-space compact unit label from a pint display unit."""
    # Currency is not a fmt_qty value-kind. It must go through fmt_usd(), which
    # owns the Pandoc-safe escaped "\$" and never emits the literal "USD".
    # Routing dollars through fmt_qty here would print a bare " USD" suffix that
    # the source currency checker cannot see (it is generated at render time).
    if str(display_unit) in {"dollar", "USD", "EUR"}:
        raise ValueError(
            "fmt_qty() does not format currency. Use fmt_usd(amount, ...) so the "
            "dollar sign is escaped for prose and no literal 'USD' is emitted."
        )
    try:
        label = f"{_coerce_unit(display_unit):~P}"
    except Exception:
        label = f"{display_unit}"
    return f" {_normalize_unit_label(label)}"
