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


# --- Great-circle distance (added 2026-10-20) --------------------------------
def calc_great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Great-circle distance between two points on a spherical Earth (haversine).

    ``d = 2 R asin(sqrt(sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2)))``
    with ``R`` the Earth's mean radius. Latitudes and longitudes are in degrees.
    Real fiber routes are longer than the great circle, so a propagation floor
    computed from this distance is a lower bound.

    Returns
    -------
    Quantity
        Distance in kilometers.
    """
    import math
    from .constants import EARTH_MEAN_RADIUS
    la1, lo1, la2, lo2 = map(math.radians, (lat1_deg, lon1_deg, lat2_deg, lon2_deg))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return (2 * math.asin(math.sqrt(h)) * EARTH_MEAN_RADIUS).to(ureg.kilometer)
