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
WATER_DENSITY = 1.0 * ureg.kilogram / ureg.liter                         # liquid water, ~1 kg per liter (added 2026-10-20)
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

# --- Earth radius for great-circle distances (added 2026-10-20) -------------
# Mean radius R1 = (2a + b) / 3 of the WGS 84 ellipsoid, the radius the
# haversine great-circle formula uses (calc_great_circle_distance).
# Source: Moritz (2000), Geodetic Reference System 1980, J. Geodesy 74, mean
# radius R1 = 6,371.0088 km; rounded to the conventional 6,371 km.
EARTH_MEAN_RADIUS = 6371.0 * ureg.kilometer
# --- Copper winding resistance (added 2026-10-20) ---------------------------
# Temperature coefficient of resistance of annealed copper, 100 percent IACS,
# stated at 20 degC: R(T) = R_20 [1 + alpha (T - 20 degC)]. A property of the
# conductor, not of any motor; winding heat budgets read it from here.
# Source: NBS Handbook 100, Copper Wire Tables (1966), alpha_20 = 0.00393 per K.
COPPER_TEMPERATURE_COEFFICIENT = 0.00393 / ureg.kelvin
COPPER_COEFFICIENT_REFERENCE_TEMPERATURE = ureg.Quantity(20.0, ureg.degC)
