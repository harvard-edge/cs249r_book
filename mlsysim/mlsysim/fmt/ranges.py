"""Range, multiple, count, token/parameter scale, rate, and ratio formatters."""

from ..core.units import ureg
from .core import (
    MarkdownStr,
    _DECIMAL_SCALE_FACTORS,
    _USD_DENOMINATORS,
    _auto_decimal_scale,
    _clean_text_atom,
    _coerce_unit,
    _denominator_suffix,
    _dimensionless_magnitude,
    _display_prefix,
    _label_suffix,
    _numeric_magnitude,
    _parse_formatted_number,
    _percent_ratio_value,
    _range_endpoint_precisions,
    _require_finite,
    _resolve_decimal_scale,
    _resolve_display_precision,
    _resolve_scaled_count_precision,
    _resolve_unit_arg,
    _validate_count_value,
    fmt,
    fmt_int,
    fmt_usd,
)
from .quantities import (
    _quantity_suffix,
    _time_word_suffix,
    fmt_sci,
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
    from ..core.units import param

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
    from ..core.units import count

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
    from ..core.units import flop

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
    from ..core.units import USD

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
