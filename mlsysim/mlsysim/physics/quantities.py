"""Quantity-first formula helpers for LEGO cells — return Pint quantities, never strings."""

from __future__ import annotations

import pint

from mlsysim.core.units import (
    Bparam,
    byte,
    count,
    gram,
    hour,
    joule,
    kilogram,
    kWh,
    param,
    second,
    ureg,
    watt,
)

__all__ = [
    "transfer_time",
    "compute_time",
    "energy_from_power",
    "carbon_from_energy",
    "memory_from_params",
    "token_throughput",
]


def _require_quantity(value, *, name: str) -> pint.UnitRegistry.Quantity:
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity.")
    return value


def transfer_time(payload, bandwidth):
    """Return duration for moving payload bytes over bandwidth."""
    payload = _require_quantity(payload, name="transfer_time payload").to(byte)
    bandwidth = _require_quantity(bandwidth, name="transfer_time bandwidth").to(
        byte / second
    )
    return (payload / bandwidth).to(second)


def compute_time(work, throughput):
    """Return duration for work divided by operation rate."""
    work = _require_quantity(work, name="compute_time work")
    throughput = _require_quantity(throughput, name="compute_time throughput")
    return (work / throughput).to(second)


def energy_from_power(power, duration):
    """Return energy for power applied over duration."""
    power = _require_quantity(power, name="energy_from_power power").to(watt)
    duration = _require_quantity(duration, name="energy_from_power duration").to(
        second
    )
    return (power * duration).to(joule)


def carbon_from_energy(energy, carbon_intensity):
    """Return carbon mass from energy and grid intensity (e.g. gram/kWh)."""
    energy = _require_quantity(energy, name="carbon_from_energy energy").to(kWh)
    carbon_intensity = _require_quantity(
        carbon_intensity, name="carbon_from_energy carbon_intensity"
    )
    if not carbon_intensity.check("[mass] / [energy]"):
        carbon_intensity = carbon_intensity.to(gram / kWh)
    return (energy * carbon_intensity).to(kilogram)


def memory_from_params(parameters, bytes_per_param):
    """Return memory footprint from parameter count and bytes per parameter."""
    parameters = _require_quantity(parameters, name="memory_from_params parameters").to(
        param
    )
    bytes_per_param = _require_quantity(
        bytes_per_param, name="memory_from_params bytes_per_param"
    ).to(byte / param)
    return (parameters * bytes_per_param).to(byte)


def token_throughput(tokens, duration):
    """Return token rate from token count and duration."""
    tokens = _require_quantity(tokens, name="token_throughput tokens").to(count)
    duration = _require_quantity(duration, name="token_throughput duration").to(second)
    return (tokens / duration).to(count / second)
