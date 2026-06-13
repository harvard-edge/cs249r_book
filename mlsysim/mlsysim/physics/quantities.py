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
    """Duration to move a payload at a sustained bandwidth: ``t = bytes / (bytes/s)``.

    Pure streaming model — no per-message latency or protocol overhead.
    ``payload`` must be byte-dimensioned, ``bandwidth`` byte-rate-dimensioned
    (any scale; Pint converts). Returns a Quantity in seconds.
    """
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
    """Energy for constant power applied over a duration: ``E = P * t``.

    ``power`` must be power-dimensioned (W/kW/MW), ``duration``
    time-dimensioned. Returns a Quantity in joules (use ``.to(kWh)`` for
    billing-scale display).
    """
    power = _require_quantity(power, name="energy_from_power power").to(watt)
    duration = _require_quantity(duration, name="energy_from_power duration").to(
        second
    )
    return (power * duration).to(joule)


def carbon_from_energy(energy, carbon_intensity):
    """Carbon mass from energy and grid carbon intensity: ``m = E * CI``.

    ``energy`` must be energy-dimensioned (converted to kWh);
    ``carbon_intensity`` must be mass-per-energy (e.g. g/kWh, the unit grid
    operators publish; coerced if needed). Returns a Quantity in kilograms
    of CO2 (the CO2/CO2e distinction is the caller's labeling concern).
    """
    energy = _require_quantity(energy, name="carbon_from_energy energy").to(kWh)
    carbon_intensity = _require_quantity(
        carbon_intensity, name="carbon_from_energy carbon_intensity"
    )
    if not carbon_intensity.check("[mass] / [energy]"):
        carbon_intensity = carbon_intensity.to(gram / kWh)
    return (energy * carbon_intensity).to(kilogram)


def memory_from_params(parameters, bytes_per_param):
    """Weight-memory footprint: ``bytes = params * bytes_per_param``.

    ``parameters`` must be param-dimensioned (e.g. ``7 * Bparam``);
    ``bytes_per_param`` is byte/param (2 for FP16/BF16, 4 for FP32, 1 for
    INT8/FP8). Weights only — excludes activations, gradients, optimizer
    state, and KV cache. Returns a Quantity in bytes.
    """
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
