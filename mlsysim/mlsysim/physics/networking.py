"""Network latency and distance physics."""

from __future__ import annotations

from mlsysim.core.units import ureg
from .constants import SPEED_OF_LIGHT_FIBER_KM_S

from ._units import _ensure_unit


def calc_network_latency_ms(distance_km):
    """Calculate the round-trip propagation floor for an optical-fiber path.

    Parameters
    ----------
    distance_km : Quantity or float
        Nonnegative one-way fiber-route length; bare numbers mean kilometers.

    Returns
    -------
    float
        Round-trip propagation duration in milliseconds.

    Notes
    -----
    Equation: RTT = 2 d/vfiber, using SPEED_OF_LIGHT_FIBER_KM_S.

    Assumptions: Both directions have the supplied route length and the registry's
    constant propagation speed.

    Excludes: Switching, queueing, serialization, transceiver processing, and route
    stretch if supplied distance is geographic.

    References
    ----------
    Derivation: distance divided by speed gives one-way travel time; the return
    path doubles it. The propagation-speed convention is sourced with
    SPEED_OF_LIGHT_FIBER_KM_S in mlsysim.physics.constants.

    Examples
    --------
    >>> calc_network_latency_ms(200) == 2 * calc_network_latency_ms(100)
    True
    """
    d = _ensure_unit(distance_km, ureg.kilometer, "distance_km")
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.m_as(ureg.millisecond)


# --- Great-circle distance (added 2026-10-20) --------------------------------
def calc_great_circle_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Calculate spherical great-circle distance between geographic coordinates.

    Parameters
    ----------
    lat1_deg : float
        First latitude in degrees, conventionally in [-90, 90].
    lon1_deg : float
        First longitude in degrees, conventionally in [-180, 180].
    lat2_deg : float
        Second latitude in degrees, conventionally in [-90, 90].
    lon2_deg : float
        Second longitude in degrees, conventionally in [-180, 180].

    Returns
    -------
    Quantity
        Shortest spherical surface distance in kilometers.

    Notes
    -----
    Equation: d = 2 R ``asin(sqrt(sin(dphi/2)**2`` + cos(phi1) cos(phi2)
    ``sin(dlambda/2)**2))``, with angles in radians and the registry mean Earth radius R.

    Assumptions: Earth is a sphere; input coordinates refer to points on its surface.

    Excludes: Ellipsoidal geodesy, terrain, cable-route detours, and network delay.

    References
    ----------
    Derivation: the haversine identity converts the central angle between
    spherical coordinates to an arc length R times angle. EARTH_MEAN_RADIUS
    is sourced in mlsysim.physics.constants.

    Examples
    --------
    >>> calc_great_circle_distance(0, 0, 0, 0).magnitude == 0
    True
    >>> calc_great_circle_distance(0, 0, 0, 90) == calc_great_circle_distance(0, 90, 0, 0)
    True
    """
    import math
    from .constants import EARTH_MEAN_RADIUS

    la1, lo1, la2, lo2 = map(math.radians, (lat1_deg, lon1_deg, lat2_deg, lon2_deg))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return (2 * math.asin(math.sqrt(h)) * EARTH_MEAN_RADIUS).to(ureg.kilometer)
