"""Assertion guards and LaTeX math-mode formatting helpers."""

from ..core.units import ureg
from .core import (
    MarkdownStr,
    _coerce_unit,
    _numeric_magnitude,
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
