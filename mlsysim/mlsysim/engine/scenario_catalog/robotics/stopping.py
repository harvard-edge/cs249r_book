"""Named robotics scenarios for stopping.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st


# --- Embodied stopping, floor traction, and domain compute (added 2026-10-20)
# The running machine's elementary stop (Embodied.MobileManipulator and
# Embodied.Scenario hold every input), the Tempe record read through the
# same budget (published record), a floor-friction drop, and
# robot-policy training compute against the language frontier.
WarehouseAisleStopAtDriveLimit = _st.StoppingBudgetScenario(
    name="Warehouse mobile manipulator stopping at its drive limit",
    description="The loaded base at its drive limit, its pre-brake delay, and its credible "
    "deceleration, against the rack-end clear distance (illustrative inputs).",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
    deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    clear_distance=_Embodied.Scenario.WarehouseAisle.d_clear,
)


WarehouseAisleStopHalfSpeed = WarehouseAisleStopAtDriveLimit.variant(
    name="Warehouse mobile manipulator stopping at half its drive limit",
    description="The drive-limit stop with the speed halved; the delay and deceleration are unchanged.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity
    / 2,  # scenario assumption: half speed
)


WarehouseInferenceHiccup = _st.StoppingBudgetScenario(
    name="A 100 ms inference stall on the warehouse base at its drive limit",
    description="The base repeats its last forward-velocity command while the policy stalls; "
    "delay_travel is the distance driven with nothing checking it.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    delay=Q_(100, "ms"),  # scenario assumption: inference stall
    deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
)


TempeActionSuppression = _st.StoppingBudgetScenario(
    name="Tempe: the one-second action suppression at the recorded speeds",
    description="NTSB HAR-19/03: emergency recognized at 43.2 mph, braking withheld for 1 s "
    "until 40.5 mph; braking from the recognition speed at the road's 0.92 g.",
    speed=Q_(43.2, "mph"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
    speed_at_delay_end=Q_(40.5, "mph"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
    delay=Q_(1.0, "s"),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
    deceleration=(0.92 * Q_(1, "standard_gravity")).to(
        "m/s**2"
    ),  # NTSB, Collision Between Vehicle Controlled by Developmental Automat...
)


HydraulicFluidSlickTraction = _st.TractionLossScenario(
    name="Hydraulic fluid on a dry concrete floor under the warehouse base",
    description="The warehouse aisle's illustrative floor friction, dry (0.60) and under a "
    "clear oil film (0.05), against the running machine's credible deceleration "
    "with all weight on the driven wheels.",
    mu_before=_Embodied.Scenario.WarehouseAisle.mu_dry,  # the aisle's dry floor (illustrative)
    mu_after=_Embodied.Scenario.WarehouseAisle.mu_oil_film,  # the aisle's oil film (illustrative)
    deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
)
