"""Network latency and distance physics."""

from __future__ import annotations

from mlsysim.core.units import ureg
from .constants import SPEED_OF_LIGHT_FIBER_KM_S

from ._units import _ensure_unit


def calc_network_latency_ms(distance_km):
    """Round-trip latency in milliseconds (speed of light in fiber, c/1.5)."""
    d = _ensure_unit(distance_km, ureg.kilometer, "distance_km")
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.m_as(ureg.millisecond)
