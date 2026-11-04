"""Quantity-first formula helpers for LEGO cells — return Pint quantities, never strings."""

from __future__ import annotations

import pint

from mlsysim.core.units import (
    byte,
    count,
    gram,
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
    """Calculate streaming duration from payload and bandwidth.

    Parameters
    ----------
    payload : Quantity
        Nonnegative payload in byte-compatible units.
    bandwidth : Quantity
        Positive sustained byte rate, such as GB/s.

    Returns
    -------
    Quantity
        Streaming duration in seconds.

    Notes
    -----
    Equation: t = payload / bandwidth.

    Assumptions: The supplied bandwidth is constant and available to this transfer.

    Excludes: Per-message latency, protocol headers, queueing, and contention.

    References
    ----------
    Derivation: Sustained bandwidth is transferred bytes per unit time. Dividing payload
    bytes by this rate cancels the byte unit and leaves transfer duration.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> transfer_time(Q_(4, "GB"), Q_(2, "GB/s")) == Q_(2, "s")
    True
    """
    payload = _require_quantity(payload, name="transfer_time payload").to(byte)
    bandwidth = _require_quantity(bandwidth, name="transfer_time bandwidth").to(byte / second)
    return (payload / bandwidth).to(second)


def compute_time(work, throughput):
    """Calculate duration from operation count and sustained rate.

    Parameters
    ----------
    work : Quantity
        Nonnegative operation count, with units compatible with throughput.
    throughput : Quantity
        Positive sustained rate of those operations per unit time.

    Returns
    -------
    Quantity
        Computation duration in seconds.

    Notes
    -----
    Equation: t = work / throughput.

    Assumptions: The supplied rate applies throughout the workload.

    Excludes: Memory limits, launch overhead, and changes in utilization.

    References
    ----------
    Derivation: Sustained throughput is completed operation count per unit time. Dividing
    the required count by this rate cancels the operation unit and leaves elapsed time.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> compute_time(Q_(4, "flop"), Q_(2, "flop/s")) == Q_(2, "s")
    True
    """
    work = _require_quantity(work, name="compute_time work")
    throughput = _require_quantity(throughput, name="compute_time throughput")
    return (work / throughput).to(second)


def energy_from_power(power, duration):
    """Integrate constant power over a duration.

    Parameters
    ----------
    power : Quantity
        Nonnegative constant power in watt-compatible units.
    duration : Quantity
        Nonnegative elapsed time.

    Returns
    -------
    Quantity
        Energy in joules.

    Notes
    -----
    Equation: E = P t.

    Assumptions: Power is constant, or represents the correct time-average for the interval.

    Excludes: Time-varying power integration and facility overhead not already in power.

    References
    ----------
    Derivation: Integrate constant power P over time: E = integral(P dt) = P t.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> energy_from_power(Q_(2, "W"), Q_(3, "s")) == Q_(6, "J")
    True
    """
    power = _require_quantity(power, name="energy_from_power power").to(watt)
    duration = _require_quantity(duration, name="energy_from_power duration").to(second)
    return (power * duration).to(joule)


def carbon_from_energy(energy, carbon_intensity):
    """Convert energy consumption to emissions mass.

    Parameters
    ----------
    energy : Quantity
        Nonnegative energy; converted internally to kWh.
    carbon_intensity : Quantity
        Nonnegative mass per unit energy, such as g/kWh; caller specifies whether mass
        denotes CO2 or CO2e.

    Returns
    -------
    Quantity
        Emissions mass in kilograms.

    Notes
    -----
    Equation: m = E CI.

    Assumptions: One intensity represents the entire energy-consumption interval.

    Excludes: Embodied emissions, temporal grid variation, and overhead outside the supplied
    energy.

    References
    ----------
    Derivation: Carbon intensity is emissions mass divided by energy. Multiplying consumed
    energy by the corresponding intensity cancels the energy unit and leaves emissions mass.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> carbon_from_energy(Q_(2, "kWh"), Q_(500, "g/kWh")) == Q_(1, "kg")
    True
    """
    energy = _require_quantity(energy, name="carbon_from_energy energy").to(kWh)
    carbon_intensity = _require_quantity(carbon_intensity, name="carbon_from_energy carbon_intensity")
    if not carbon_intensity.check("[mass] / [energy]"):
        carbon_intensity = carbon_intensity.to(gram / kWh)
    return (energy * carbon_intensity).to(kilogram)


def memory_from_params(parameters, bytes_per_param):
    """Convert parameter count and storage width to weight bytes.

    Parameters
    ----------
    parameters : Quantity
        Nonnegative parameter count in param-compatible units.
    bytes_per_param : Quantity
        Nonnegative storage width in byte/param units.

    Returns
    -------
    Quantity
        Weight storage in bytes.

    Notes
    -----
    Equation: M = P b.

    Assumptions: Every parameter has the same supplied storage width.

    Excludes: Activations, gradients, optimizer state, KV cache, metadata, and alignment.

    References
    ----------
    Derivation: Each stored parameter occupies the supplied byte width. Multiplying the
    parameter count by bytes per parameter cancels the parameter unit and leaves stored
    bytes.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> memory_from_params(Q_(3, "param"), Q_(2, "byte/param")) == Q_(6, "byte")
    True
    """
    parameters = _require_quantity(parameters, name="memory_from_params parameters").to(param)
    bytes_per_param = _require_quantity(bytes_per_param, name="memory_from_params bytes_per_param").to(byte / param)
    return (parameters * bytes_per_param).to(byte)


def token_throughput(tokens, duration):
    """Calculate average token rate over an interval.

    Parameters
    ----------
    tokens : Quantity
        Nonnegative token count in count-compatible units.
    duration : Quantity
        Positive elapsed time.

    Returns
    -------
    Quantity
        Average rate in count/second.

    Notes
    -----
    Equation: r = N_tokens / t.

    Assumptions: The count and interval describe the same work; the caller defines prompt or
    output tokens.

    Excludes: Instantaneous rate, latency quantiles, and separation of prefill from decode.

    References
    ----------
    Derivation: Average throughput is the count of completed items divided by their
    observation interval. Using token count and elapsed time gives tokens per second.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> token_throughput(Q_(6, "count"), Q_(2, "s")) == Q_(3, "count/s")
    True
    """
    tokens = _require_quantity(tokens, name="token_throughput tokens").to(count)
    duration = _require_quantity(duration, name="token_throughput duration").to(second)
    return (tokens / duration).to(count / second)
