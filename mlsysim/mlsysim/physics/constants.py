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

# --- Thermophysical properties (added 2026-09-28) ---------------------------
# Standard engineering-handbook values near 25 degC and sea-level pressure
# (e.g. Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, property
# tables). Properties of the physical media, not of any product.
AIR_SPECIFIC_HEAT = 1005 * ureg.joule / (ureg.kilogram * ureg.kelvin)    # c_p of dry air
AIR_DENSITY_SEA_LEVEL = 1.2 * ureg.kilogram / ureg.meter ** 3             # dry air, ~20 degC, 1 atm
WATER_SPECIFIC_HEAT = 4180 * ureg.joule / (ureg.kilogram * ureg.kelvin)  # liquid water
THERMAL_CONDUCTIVITY_AIR = 0.026 * ureg.watt / (ureg.meter * ureg.kelvin)
THERMAL_CONDUCTIVITY_WATER = 0.6 * ureg.watt / (ureg.meter * ureg.kelvin)
THERMAL_CONDUCTIVITY_COPPER = 400 * ureg.watt / (ureg.meter * ureg.kelvin)

# --- Insulation thermal class (added 2026-09-29) ----------------------------
# Class F (155) of IEC 60085:2007, "Electrical insulation - Thermal evaluation
# and designation": the hot-spot temperature at which an electrical insulation
# system reaches its rated life. A rating of the insulation standard, not of any
# product; motor data sheets state the class, and the book's thermal budgets
# (vol4 body and training chapters) read the limit from here.
# Source: IEC 60085:2007, Table 1 (thermal class 155, letter designation F).
INSULATION_CLASS_F_LIMIT = ureg.Quantity(155.0, ureg.degC)
