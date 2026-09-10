"""Shared unit helpers for physics formulas."""

from __future__ import annotations

import pint

from mlsysim.core.units import ureg


def _ensure_unit(val, expected_unit, param_name="Value"):
    """
    Attach a unit if val is a raw number; verify dimensional correctness AND
    convert to ``expected_unit`` if it is already a Pint Quantity.

    Returning the *converted* quantity (audit fix 2026-06-06; previously the
    quantity was returned unconverted) makes a downstream ``.magnitude`` read
    safe: a caller that passes ``Q_("120 1/minute")`` where Hz is expected now
    yields magnitude 2.0, not a silent 60x misread.
    """
    if isinstance(val, (int, float)):
        return val * expected_unit
    if isinstance(val, ureg.Quantity):
        if not val.check(expected_unit):
            raise pint.DimensionalityError(
                val.units,
                expected_unit,
                extra_msg=(
                    f"\n[Pedagogical Error] '{param_name}' requires units of "
                    f"{expected_unit.dimensionality}. You provided '{val.units}'."
                ),
            )
        return val.to(expected_unit)
    raise TypeError(
        f"Expected a number or Pint Quantity for {param_name}, got {type(val).__name__}"
    )
