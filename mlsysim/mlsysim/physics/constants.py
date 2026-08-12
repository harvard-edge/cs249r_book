"""Universal physical constants (the physics layer's ground truth).

These are genuine constants of nature / physical media — not hardware specs,
model specs, or tunable knobs — so they live with the laws in ``physics/``
rather than in ``core/constants.py`` (which is now units-only). See
the project MLSysIM rules -> Canonical organization (decision step 2).
"""
from ..core.units import ureg, second

# Light in optical fiber propagates at ~2/3 c: silica's refractive index of
# ~1.5 gives ~200,000 km/s, the standard teaching value. The round-trip
# latency floor of any fiber link derives from this.
SPEED_OF_LIGHT_FIBER_KM_S = 200_000 * ureg.kilometer / second
