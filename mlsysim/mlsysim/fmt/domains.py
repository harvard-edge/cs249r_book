"""Domain-specific physical quantity formatters and automatic unit selectors."""

from ..core.units import ureg
from .core import (
    _coerce_unit,
    _resolve_display_precision,
)
from .quantities import (
    fmt_qty,
    fmt_time,
)
from .ranges import (
    fmt_rate,
)


def _pick_length_unit(qty):
    """Auto-select meter vs kilometer for prose length display."""
    from ..core.units import kilometer, meter

    q = qty.to(meter)
    mag = abs(q.magnitude)
    if mag < 1000:
        return meter
    return kilometer


def _pick_area_unit(qty):
    """Auto-select square millimeters for chip-scale areas, else square meters."""
    from ..core.units import meter

    q = qty.to(meter**2)
    if abs(q.magnitude) < 0.01:
        return ureg.millimeter**2
    return meter**2


def _pick_memory_unit(qty, *, binary=False):
    from ..core.units import GB, GiB, KB, KiB, MB, MiB, PB, TB, TiB, byte

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
    from ..core.units import GB, MB, TB, byte, second

    q = qty.to(byte / second)
    mag = abs(q.magnitude)
    if mag < 1e9:
        return MB / second
    if mag < 1e12:
        return GB / second
    return TB / second


def _pick_flop_rate_unit(qty):
    from ..core.units import EFLOP, GFLOP, PFLOP, TFLOP, ZFLOP, second

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
    from ..core.units import EFLOP, GFLOP, KFLOP, MFLOP, PFLOP, TFLOP, ZFLOP, flop

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
    from ..core.units import GOPS, KOPS, MOPS, OPS, TOPS

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
    from ..core.units import GW, MW, kilowatt, watt

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
    from ..core.units import GWh, MWh, Wh, joule, kWh

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
    from ..core.units import gram, kilogram, metric_ton

    q = qty.to(gram)
    mag = abs(q.magnitude)
    if mag < 1e3:
        return gram
    kg = q.to(kilogram).magnitude
    if kg < 1e3:
        return kilogram
    return metric_ton


def _pick_mass_unit(qty):
    """Auto-select gram, kilogram, or metric ton for prose mass display."""
    from ..core.units import gram, kilogram, metric_ton

    q = qty.to(gram)
    mag = abs(q.magnitude)
    if mag < 1e3:
        return gram
    kg = q.to(kilogram).magnitude
    if kg < 1e3:
        return kilogram
    return metric_ton


def _pick_carbon_intensity_unit(qty):
    from ..core.units import gram, kWh

    # Grid intensities are normally communicated as g/kWh. Callers
    # can force kg/kWh when prose needs that convention.
    qty.to(gram / kWh)
    return gram / kWh


def _pick_latency_unit(qty):
    from ..core.units import hour, microsecond, millisecond, minute, nanosecond, second

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


def _pick_frequency_unit(qty):
    mag = abs(qty.to(ureg.hertz).magnitude)
    if mag >= 1e9:
        return ureg.gigahertz
    if mag >= 1e6:
        return ureg.megahertz
    if mag >= 1e3:
        return ureg.kilohertz
    return ureg.hertz



def fmt_power(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    per=None,
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
        per=per,
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
    per=None,
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
        per=per,
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
    per=None,
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
        per=per,
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
    per=None,
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
        per=per,
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
    from ..core.units import byte, flop

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


def _temperature_unit_label(display_unit):
    unit_str = str(display_unit)
    if unit_str in ("K", "kelvin"):
        return "K"
    if "°C" in unit_str or "degC" in unit_str or "celsius" in unit_str.lower():
        return "°C"
    if "°F" in unit_str or "degF" in unit_str or "fahrenheit" in unit_str.lower():
        return "°F"
    return None


def _temperature_rate_unit_label(display_unit):
    unit_str = str(display_unit)
    if "K" in unit_str or "kelvin" in unit_str:
        return "K/s"
    if "°C" in unit_str or "degC" in unit_str or "celsius" in unit_str.lower():
        return "°C/s"
    if "°F" in unit_str or "degF" in unit_str or "fahrenheit" in unit_str.lower():
        return "°F/s"
    return None


def fmt_temperature(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format absolute temperatures, defaulting to degrees Celsius (or Kelvin if input is in K)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_temperature() requires a Pint Quantity.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        q_units = str(quantity.units)
        display_unit = ureg.kelvin if ("kelvin" in q_units or q_units == "K") else ureg.degC
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
        unit_label=_temperature_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_temperature_rate(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format temperature-change rates, defaulting to degrees Celsius per second (or K/s)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_temperature_rate() requires a Pint Quantity.")
    from ..core.units import second

    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        q_units = str(quantity.units)
        display_unit = (ureg.kelvin / second) if ("kelvin" in q_units or "K" in q_units) else (ureg.delta_degC / second)
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
        unit_label=_temperature_rate_unit_label(display_unit),
        trim_trailing_zeros=auto_precision,
    )


def fmt_compute_efficiency(quantity, *, unit=None, precision=None, commas=False):
    """Format accelerator efficiency as FLOP throughput per watt."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_compute_efficiency() requires a Pint Quantity.")
    from ..core.units import TFLOP, second, watt

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
    from ..core.units import byte, pJ

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
    from ..core.units import bit, pJ

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
    from ..core.units import flop, pJ

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
    from ..core.units import flop, pJ

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
    from ..core.units import meter

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
    from ..core.units import meter

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
    from ..core.units import watt

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
    from ..core.units import J, K, kg

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


def fmt_memory(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    binary=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    per=None,
):
    """Auto-scale memory sizes for prose."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_memory() requires a Pint Quantity.")
    display_unit = (
        _coerce_unit(unit) if unit is not None else _pick_memory_unit(quantity, binary=binary)
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
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        per=per,
        trim_trailing_zeros=auto_precision,
    )


def _memory_capacity_unit_label(display_unit):
    """Return vendor-style capacity labels for binary memory display units."""
    from ..core.units import GiB, KiB, MiB, TiB

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
    from ..core.units import L, day, hour, kWh

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
    from ..core.units import L

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
    from ..core.units import L, day

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
    from ..core.units import L, kWh

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


def fmt_frequency(quantity, *, unit=None, precision=None, commas=False):
    """Format frequency quantities for prose (Hz, kHz, MHz, GHz)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_frequency() requires a Pint Quantity.")
    if (1 * quantity).dimensionality != (1 * ureg.hertz).dimensionality:
        raise ValueError(f"fmt_frequency unit must have frequency dimensionality (1/s), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_frequency_unit(quantity)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_acceleration(quantity, *, unit=None, precision=None, commas=False):
    """Format acceleration quantities for prose (m/s²)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_acceleration() requires a Pint Quantity.")
    accel_dim = (1 * (ureg.meter / (ureg.second**2))).dimensionality
    if (1 * quantity).dimensionality != accel_dim:
        raise ValueError(f"fmt_acceleration unit must have acceleration dimensionality (m/s²), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.meter / (ureg.second**2))
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    # 2026-10-13: multiples of standard gravity print as "g", not pint's "g_0"
    # (replaces hand-typed suffix=" g" after dividing by ureg.standard_gravity).
    unit_label = "g" if display_unit == ureg.standard_gravity else None
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label=unit_label,
                   trim_trailing_zeros=auto_precision)


def fmt_torque(quantity, *, unit=None, precision=None, commas=False):
    """Format torque/moment quantities for prose (N·m)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_torque() requires a Pint Quantity.")
    torque_dim = (1 * (ureg.newton * ureg.meter)).dimensionality
    if (1 * quantity).dimensionality != torque_dim:
        raise ValueError(f"fmt_torque unit must have torque dimensionality (N·m), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.newton * ureg.meter)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="N·m",
        trim_trailing_zeros=auto_precision,
    )


def fmt_torque_rate(quantity, *, unit=None, precision=None, commas=False):
    """Format torque-rate quantities for prose (N·m/s), e.g. a steering-torque ramp."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_torque_rate() requires a Pint Quantity.")
    rate_dim = (1 * (ureg.newton * ureg.meter / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != rate_dim:
        raise ValueError(f"fmt_torque_rate unit must have torque-rate dimensionality (N·m/s), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.newton * ureg.meter / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="N·m/s",
        trim_trailing_zeros=auto_precision,
    )


def fmt_torque_constant(quantity, *, unit=None, precision=None, commas=False):
    """Format a motor torque constant for prose (N·m/A)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_torque_constant() requires a Pint Quantity.")
    kt_dim = (1 * (ureg.newton * ureg.meter / ureg.ampere)).dimensionality
    if (1 * quantity).dimensionality != kt_dim:
        raise ValueError(f"fmt_torque_constant unit must have N·m/A dimensionality, got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.newton * ureg.meter / ureg.ampere)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="N·m/A",
        trim_trailing_zeros=auto_precision,
    )


def fmt_motor_constant(quantity, *, unit=None, precision=None, commas=False):
    """Format a motor constant K_m = K_t / sqrt(R) for prose (N·m/√W)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_motor_constant() requires a Pint Quantity.")
    km_unit = ureg.newton * ureg.meter / ureg.watt**0.5
    if (1 * quantity).dimensionality != (1 * km_unit).dimensionality:
        raise ValueError(f"fmt_motor_constant unit must have N·m/√W dimensionality, got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else km_unit
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(
        q,
        display_unit,
        precision=p,
        commas=commas,
        unit_label="N·m/√W",
        trim_trailing_zeros=auto_precision,
    )


def fmt_force(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format force quantities for prose (N, kN, mN)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_force() requires a Pint Quantity.")
    force_dim = (1 * ureg.newton).dimensionality
    if (1 * quantity).dimensionality != force_dim:
        raise ValueError(f"fmt_force unit must have force dimensionality (N), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.newton
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = "N" if display_unit == ureg.newton else None
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


def fmt_force_rate(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format force-rate quantities for prose (N/s, N/ms), e.g. braking or clamp ramps."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_force_rate() requires a Pint Quantity.")
    rate_dim = (1 * (ureg.newton / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != rate_dim:
        raise ValueError(f"fmt_force_rate unit must have force-rate dimensionality (N/s), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.newton / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = None
    try:
        if abs((1 * display_unit).to(ureg.newton / ureg.second).magnitude - 1) < 1e-12:
            unit_label = "N/s"
        elif abs((1 * display_unit).to(ureg.newton / ureg.millisecond).magnitude - 1) < 1e-12:
            unit_label = "N/ms"
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


def fmt_mass(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Auto-scale mass quantities for prose (g, kg, t)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_mass() requires a Pint Quantity.")
    mass_dim = (1 * ureg.kilogram).dimensionality
    if (1 * quantity).dimensionality != mass_dim:
        raise ValueError(f"fmt_mass unit must have mass dimensionality, got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else _pick_mass_unit(quantity)
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


def fmt_angular_velocity(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format angular velocity quantities for prose (rad/s, deg/s, rpm)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_angular_velocity() requires a Pint Quantity.")
    ang_vel_dim = (1 * (ureg.radian / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != ang_vel_dim:
        raise ValueError(f"fmt_angular_velocity unit must have angular velocity dimensionality (angle/time), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.radian / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = None
    try:
        if abs((1 * display_unit).to(ureg.radian / ureg.second).magnitude - 1) < 1e-12:
            unit_label = "rad/s"
        elif abs((1 * display_unit).to(ureg.degree / ureg.second).magnitude - 1) < 1e-12:
            unit_label = "deg/s"
        elif str(display_unit) in {"rpm", "revolutions_per_minute"} or display_unit == ureg.Unit("rpm"):
            unit_label = "rpm"
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
def fmt_angular_acceleration(quantity, *, unit=None, precision=None, commas=False):
    """Format angular acceleration quantities for prose (rad/s², deg/s²).

    Added 2026-10-13: joint-acceleration limits were rendered with a
    hand-typed ``suffix=" rad/s²"``; this keeps the unit on the Quantity.
    """
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_angular_acceleration() requires a Pint Quantity.")
    ang_acc_dim = (1 * (ureg.radian / ureg.second**2)).dimensionality
    if (1 * quantity).dimensionality != ang_acc_dim:
        raise ValueError(
            f"fmt_angular_acceleration unit must have angular acceleration dimensionality (angle/time²), got {quantity}."
        )
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.radian / ureg.second**2)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = None
    if abs((1 * display_unit).to(ureg.radian / ureg.second**2).magnitude - 1) < 1e-12:
        unit_label = "rad/s²"
    elif abs((1 * display_unit).to(ureg.degree / ureg.second**2).magnitude - 1) < 1e-12:
        unit_label = "deg/s²"
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label=unit_label,
                   trim_trailing_zeros=auto_precision)


def fmt_velocity(quantity, *, unit=None, precision=None, commas=False):
    """Format velocity/speed quantities for prose (m/s, km/h, mph)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_velocity() requires a Pint Quantity.")
    vel_dim = (1 * (ureg.meter / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != vel_dim:
        raise ValueError(f"fmt_velocity unit must have velocity dimensionality (length/time), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.meter / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = None
    try:
        if abs((1 * display_unit).to(ureg.meter / ureg.second).magnitude - 1) < 1e-12:
            unit_label = "m/s"
        elif abs((1 * display_unit).to(ureg.kilometer / ureg.hour).magnitude - 1) < 1e-12:
            unit_label = "km/h"
    except Exception:
        unit_label = None
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label=unit_label, trim_trailing_zeros=auto_precision)


def fmt_jerk(quantity, *, unit=None, precision=None, commas=False):
    """Format kinematic jerk quantities for prose (m/s³)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_jerk() requires a Pint Quantity.")
    jerk_dim = (1 * (ureg.meter / (ureg.second**3))).dimensionality
    if (1 * quantity).dimensionality != jerk_dim:
        raise ValueError(f"fmt_jerk unit must have jerk dimensionality (m/s³), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.meter / (ureg.second**3))
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label="m/s³", trim_trailing_zeros=auto_precision)


def fmt_inertia(quantity, *, unit=None, precision=None, commas=False):
    """Format moment of inertia quantities for prose (kg·m²)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_inertia() requires a Pint Quantity.")
    inertia_dim = (1 * (ureg.kilogram * (ureg.meter**2))).dimensionality
    if (1 * quantity).dimensionality != inertia_dim:
        raise ValueError(f"fmt_inertia unit must have moment of inertia dimensionality (kg·m²), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.kilogram * (ureg.meter**2))
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label="kg·m²", trim_trailing_zeros=auto_precision)


def fmt_voltage(quantity, *, unit=None, precision=None, commas=False):
    """Format electrical potential / voltage quantities for prose (V, mV, kV)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_voltage() requires a Pint Quantity.")
    volt_dim = (1 * ureg.volt).dimensionality
    if (1 * quantity).dimensionality != volt_dim:
        raise ValueError(f"fmt_voltage unit must have voltage dimensionality (V), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.volt
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_current(quantity, *, unit=None, precision=None, commas=False):
    """Format electrical current quantities for prose (A, mA)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_current() requires a Pint Quantity.")
    curr_dim = (1 * ureg.ampere).dimensionality
    if (1 * quantity).dimensionality != curr_dim:
        raise ValueError(f"fmt_current unit must have electric current dimensionality (A), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.ampere
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, trim_trailing_zeros=auto_precision)


def fmt_resistance(quantity, *, unit=None, precision=None, commas=False):
    """Format electrical resistance quantities for prose (Ω, mΩ)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_resistance() requires a Pint Quantity.")
    res_dim = (1 * ureg.ohm).dimensionality
    if (1 * quantity).dimensionality != res_dim:
        raise ValueError(f"fmt_resistance unit must have electrical resistance dimensionality (Ω), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.ohm
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    return fmt_qty(q, display_unit, precision=p, commas=commas, unit_label="Ω" if display_unit == ureg.ohm else None, trim_trailing_zeros=auto_precision)



def fmt_token_rate(
    value,
    *,
    unit="tokens/s",
    precision=0,
    commas=True,
    scale=None,
    approx=False,
    lower_bound=False,
    upper_bound=False,
    allow_negative=False,
):
    """Format an LLM token generation or ingestion rate (tokens/s)."""
    return fmt_rate(
        value,
        unit=unit,
        precision=precision,
        commas=commas,
        scale=scale,
        approx=approx,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        allow_negative=allow_negative,
    )


def fmt_thermal_resistance(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format thermal resistance quantities for prose (K/W, °C/W)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_thermal_resistance() requires a Pint Quantity.")
    res_dim = (1 * (ureg.kelvin / ureg.watt)).dimensionality
    if (1 * quantity).dimensionality != res_dim:
        raise ValueError(f"fmt_thermal_resistance unit must have thermal resistance dimensionality (temperature/power), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.kelvin / ureg.watt)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "°C" in unit_str or "degC" in unit_str:
        unit_label = "°C/W"
    else:
        unit_label = "K/W"
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


def fmt_heat_capacity(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format heat capacity quantities for prose (J/K, kJ/K)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_heat_capacity() requires a Pint Quantity.")
    cap_dim = (1 * (ureg.joule / ureg.kelvin)).dimensionality
    if (1 * quantity).dimensionality != cap_dim:
        raise ValueError(f"fmt_heat_capacity unit must have heat capacity dimensionality (energy/temperature), got {quantity}.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        base_mag = abs(quantity.to(ureg.joule / ureg.kelvin).magnitude)
        display_unit = (ureg.kilojoule / ureg.kelvin) if base_mag >= 1000.0 else (ureg.joule / ureg.kelvin)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "kilo" in unit_str or "kJ" in unit_str:
        unit_label = "kJ/K"
    else:
        unit_label = "J/K"
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


def fmt_density(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format density quantities for prose (kg/m³, g/cm³)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_density() requires a Pint Quantity.")
    density_dim = (1 * (ureg.kilogram / (ureg.meter**3))).dimensionality
    if (1 * quantity).dimensionality != density_dim:
        raise ValueError(f"fmt_density unit must have density dimensionality (mass/volume), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.kilogram / (ureg.meter**3))
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = "kg/m³" if display_unit == (ureg.kilogram / (ureg.meter**3)) else None
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


def fmt_mass_flow(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format mass flow rate quantities for prose (kg/s, kg/h)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_mass_flow() requires a Pint Quantity.")
    flow_dim = (1 * (ureg.kilogram / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != flow_dim:
        raise ValueError(f"fmt_mass_flow unit must have mass flow dimensionality (mass/time), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.kilogram / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = "kg/s" if display_unit == (ureg.kilogram / ureg.second) else None
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


fmt_mass_flow_rate = fmt_mass_flow


def fmt_volumetric_flow(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format volumetric flow rate quantities for prose (m³/s, L/s)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_volumetric_flow() requires a Pint Quantity.")
    flow_dim = (1 * ((ureg.meter**3) / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != flow_dim:
        raise ValueError(f"fmt_volumetric_flow unit must have volumetric flow dimensionality (volume/time), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ((ureg.meter**3) / ureg.second)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_label = "m³/s" if display_unit == ((ureg.meter**3) / ureg.second) else None
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


fmt_flow_rate = fmt_volumetric_flow


def fmt_charge(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format electric charge or battery capacity for prose (mAh, Ah, C)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_charge() requires a Pint Quantity.")
    charge_dim = (1 * (ureg.ampere * ureg.second)).dimensionality
    if (1 * quantity).dimensionality != charge_dim:
        raise ValueError(f"fmt_charge unit must have electric charge dimensionality (current·time), got {quantity}.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        mag_mah = abs(quantity.to(ureg.milliampere_hour).magnitude)
        display_unit = ureg.ampere_hour if mag_mah >= 1000.0 else ureg.milliampere_hour
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "milliampere_hour" in unit_str or "mAh" in unit_str:
        unit_label = "mAh"
    elif "ampere_hour" in unit_str or "Ah" in unit_str:
        unit_label = "Ah"
    elif "coulomb" in unit_str or "C" in unit_str:
        unit_label = "C"
    else:
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


fmt_battery_capacity = fmt_charge


def fmt_angle(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format planar angle quantities for prose (rad, mrad, deg)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_angle() requires a Pint Quantity.")
    try:
        quantity.to(ureg.radian)
    except Exception:
        raise ValueError(f"fmt_angle unit must be an angle (rad, deg), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else ureg.radian
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if unit_str in ("radian", "rad"):
        unit_label = "rad"
    elif unit_str in ("milliradian", "mrad"):
        unit_label = "mrad"
    elif unit_str in ("degree", "deg"):
        unit_label = "deg"
    else:
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


def fmt_inductance(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format electrical inductance quantities for prose (µH, mH, H)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_inductance() requires a Pint Quantity.")
    ind_dim = (1 * ureg.henry).dimensionality
    if (1 * quantity).dimensionality != ind_dim:
        raise ValueError(f"fmt_inductance unit must have inductance dimensionality (H), got {quantity}.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        mag_h = abs(quantity.to(ureg.henry).magnitude)
        if mag_h < 1e-3:
            display_unit = ureg.microhenry
        elif mag_h < 1.0:
            display_unit = ureg.millihenry
        else:
            display_unit = ureg.henry
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "microhenry" in unit_str or "μH" in unit_str or "µH" in unit_str or "uH" in unit_str:
        unit_label = "μH"
    elif "millihenry" in unit_str or "mH" in unit_str:
        unit_label = "mH"
    elif "henry" in unit_str or unit_str == "H":
        unit_label = "H"
    else:
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


def fmt_capacitance(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format electrical capacitance quantities for prose (pF, nF, μF, mF, F)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_capacitance() requires a Pint Quantity.")
    cap_dim = (1 * ureg.farad).dimensionality
    if (1 * quantity).dimensionality != cap_dim:
        raise ValueError(f"fmt_capacitance unit must have capacitance dimensionality (F), got {quantity}.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        mag_f = abs(quantity.to(ureg.farad).magnitude)
        if mag_f < 1e-6:
            display_unit = ureg.nanofarad
        elif mag_f < 1e-3:
            display_unit = ureg.microfarad
        elif mag_f < 1.0:
            display_unit = ureg.millifarad
        else:
            display_unit = ureg.farad
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "picofarad" in unit_str or "pF" in unit_str:
        unit_label = "pF"
    elif "nanofarad" in unit_str or "nF" in unit_str:
        unit_label = "nF"
    elif "microfarad" in unit_str or "μF" in unit_str or "µF" in unit_str or "uF" in unit_str:
        unit_label = "μF"
    elif "millifarad" in unit_str or "mF" in unit_str:
        unit_label = "mF"
    elif "farad" in unit_str or unit_str == "F":
        unit_label = "F"
    else:
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


def fmt_power_rate(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format power ramp rate quantities for prose (W/s, kW/s, MW/s, GW/s)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_power_rate() requires a Pint Quantity.")
    rate_dim = (1 * (ureg.watt / ureg.second)).dimensionality
    if (1 * quantity).dimensionality != rate_dim:
        raise ValueError(f"fmt_power_rate unit must have power ramp rate dimensionality (power/time), got {quantity}.")
    if unit is not None:
        display_unit = _coerce_unit(unit)
    else:
        mag_w = abs(quantity.to(ureg.watt / ureg.second).magnitude)
        if mag_w >= 1e9:
            display_unit = ureg.gigawatt / ureg.second
        elif mag_w >= 1e6:
            display_unit = ureg.megawatt / ureg.second
        elif mag_w >= 1e3:
            display_unit = ureg.kilowatt / ureg.second
        else:
            display_unit = ureg.watt / ureg.second
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "gigawatt" in unit_str or "GW" in unit_str:
        unit_label = "GW/s"
    elif "megawatt" in unit_str or "MW" in unit_str:
        unit_label = "MW/s"
    elif "kilowatt" in unit_str or "kW" in unit_str:
        unit_label = "kW/s"
    elif "watt" in unit_str or "W" in unit_str:
        unit_label = "W/s"
    else:
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


def fmt_stiffness(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format mechanical stiffness quantities for prose (N/m, N/mm)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_stiffness() requires a Pint Quantity.")
    stiff_dim = (1 * (ureg.newton / ureg.meter)).dimensionality
    if (1 * quantity).dimensionality != stiff_dim:
        raise ValueError(f"fmt_stiffness unit must have stiffness dimensionality (force/length), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.newton / ureg.meter)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    # 2026-09-29: the label was chosen by substring, so kN/m values printed
    # as "N/m". Build the label from the numerator and denominator prefixes.
    force_prefixes = (("meganewton", "MN"), ("kilonewton", "kN"), ("millinewton", "mN"), ("newton", "N"))
    length_prefixes = (("kilometer", "km"), ("millimeter", "mm"), ("micrometer", "µm"), ("centimeter", "cm"), ("meter", "m"))
    parts = dict(display_unit._units)
    num = " ".join(k for k, v in parts.items() if v > 0)
    den = " ".join(k for k, v in parts.items() if v < 0)
    f_lbl = next((lbl for name, lbl in force_prefixes if name in num), None)
    l_lbl = next((lbl for name, lbl in length_prefixes if name in den), None)
    if f_lbl is None or l_lbl is None:
        raise ValueError(f"fmt_stiffness cannot label unit {display_unit}; pass a force/length unit.")
    unit_label = f"{f_lbl}/{l_lbl}"
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


def fmt_volume(
    quantity,
    *,
    unit=None,
    precision=None,
    commas=False,
    approx=False,
    lower_bound=False,
    upper_bound=False,
):
    """Format 3D volume quantities for prose (m³, L, mL, cm³)."""
    if not isinstance(quantity, ureg.Quantity):
        raise TypeError("fmt_volume() requires a Pint Quantity.")
    vol_dim = (1 * (ureg.meter**3)).dimensionality
    if (1 * quantity).dimensionality != vol_dim:
        raise ValueError(f"fmt_volume unit must have volume dimensionality (length³), got {quantity}.")
    display_unit = _coerce_unit(unit) if unit is not None else (ureg.meter**3)
    q = quantity.to(display_unit)
    auto_precision = precision is None
    p = _resolve_display_precision(q.magnitude, precision)
    unit_str = str(display_unit)
    if "liter" in unit_str or unit_str == "L":
        unit_label = "L"
    elif "milliliter" in unit_str or "mL" in unit_str:
        unit_label = "mL"
    elif "centimeter" in unit_str or "cm" in unit_str:
        unit_label = "cm³"
    else:
        unit_label = "m³"
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
