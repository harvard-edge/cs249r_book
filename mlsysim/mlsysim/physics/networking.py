"""Network latency and distance physics."""

from __future__ import annotations

from mlsysim.core.units import ureg
from .constants import SPEED_OF_LIGHT_FIBER_KM_S

from ._units import _ensure_unit


def calc_network_latency_ms(distance_km):
    """Physical round-trip latency floor for a fiber link.

    Models only signal propagation: ``RTT = 2 * distance / v_fiber``, where
    ``v_fiber`` is the speed of light in optical fiber (refractive index ~1.5
    gives ~200,000 km/s, the standard teaching value; see
    ``SPEED_OF_LIGHT_FIBER_KM_S``). Real links add switching, queueing, and
    serialization delay on top, so this is a lower bound.

    Parameters
    ----------
    distance_km : Quantity or float
        One-way fiber distance; bare floats are interpreted as kilometers.

    Returns
    -------
    float
        Round-trip propagation latency in milliseconds (~1 ms per 100 km
        of one-way distance).
    """
    d = _ensure_unit(distance_km, ureg.kilometer, "distance_km")
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.m_as(ureg.millisecond)
