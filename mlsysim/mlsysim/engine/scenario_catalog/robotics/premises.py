"""Named robotics scenarios for premises.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....physics.drivetrain import calc_traction_deceleration_limit as _traction_limit
from ..robotics.stopping_enforcement import WarehouseFinishedStopAtAisleSpeed
from ..statistics.fault_injection import WarehouseFinishedStopRestricted


# --- Premises no sensor reports (added 2026-10-20) ---
# The aisle budget on a clear oil film, the film detector's range and closure test,
# the restricted transit's cost, a hypothetical packaging cell, the coverage bill of
# a shielded hazard claim, and the series spring at the door latch.
WarehouseFinishedStopOilFilm = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Warehouse base's finished stopping budget on a clear oil film",
    description="Deceleration capped by the film's friction, all weight on the driven wheels; the jerk "
    "bound belongs to the load and is unchanged.",
    deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_oil_film),
)


WarehouseFilmDetectorRange = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Range a film detector needs at the aisle speed",
    description="The dry-floor finished budget plus the travel during the detector's sensing and "
    "inference (added_delay): the stop then ends before the film's leading edge.",
    added_delay=Q_(100.0, "ms"),  # scenario assumption: chosen film-detector latency
)


WarehouseFilmDetectorClosure = _st.ZeroFailureScenario(
    name="Seeded-film closure test for the friction register entry",
    description="Clean trials that bound the detector's miss rate at the target, and the bound a "
    "smaller 1,000-trial campaign supports.",
    trials=1000,  # scenario assumption: a smaller zero-miss campaign, for contrast
    confidence=0.95,  # scenario assumption: chosen one-sided level
    target_rate=0.001,  # scenario assumption: chosen miss-rate bound for closure
)


WarehouseRestrictedTransit = _st.RestrictedTransitScenario(
    name="The aisle leg at the restricted speed on inspected floors",
    description="Speed cut and leg time when the register holds the base to the restricted mode.",
    site=_Embodied.Scenario.WarehouseAisle,
    restricted_speed=WarehouseFinishedStopRestricted.speed,
)


PackagingCellTransit = _st.PackagingCellScenario(
    name="A hypothetical packaging cell's carriage at its nominal speed",
    description="Camera-triggered stop before a fixed stop line: four delay allocations, a braking "
    "lower bound, and a reserve held back from the clearance. Not measurements.",
    mass=Q_(12.0, "kg"),  # scenario assumption: total moving mass, including load
    speed=Q_(1.5, "m/s"),  # scenario assumption: nominal speed
    clearance=Q_(60.0, "mm"),  # scenario assumption: clearance to the stop line
    reserve=Q_(15.0, "mm"),  # scenario assumption: allocated reserve
    deceleration=Q_(15.0, "m/s**2"),  # scenario assumption: braking lower bound after delay
    exposure=Q_(16.0, "ms"),  # scenario assumption: event to usable frame
    processing=Q_(36.0, "ms"),  # scenario assumption: transfer, inference, scheduling, admission
    bus=Q_(2.0, "ms"),  # scenario assumption: command to brake-driver delivery
    onset=Q_(10.0, "ms"),  # scenario assumption: delivery to full deceleration
    check_rate=Q_(1.0, "kHz"),  # scenario assumption: independent kinematic-check rate
    force_limit=Q_(220.0, "N"),  # scenario assumption: limiter setting, not material strength
    temperature_trip=Q_(105.0, "degC"),  # scenario assumption: selected sensor threshold
    stall_timeout=Q_(400.0, "ms"),  # scenario assumption: selected stall-detection timeout
    bus_trip=Q_(60.0, "V"),  # scenario assumption: inverter overvoltage threshold
)


PackagingCellRestrictedMode = PackagingCellTransit.variant(
    name="The packaging cell's carriage in the candidate restricted mode",
    description="The same chain and brake at a speed chosen below the ceiling.",
    speed=Q_(0.5, "m/s"),  # scenario assumption: restricted speed
)


PackagingCellHalfProcessing = PackagingCellTransit.variant(
    name="The packaging cell with its processing allocation halved",
    description="Sensitivity of the ceiling to the transfer-through-admission allocation.",
    processing=(PackagingCellTransit.processing / 2).to("ms"),
)


PackagingCellHalfBrake = PackagingCellTransit.variant(
    name="The packaging cell with its braking bound halved",
    description="Sensitivity of the ceiling to the qualified deceleration.",
    deceleration=(PackagingCellTransit.deceleration / 2).to("m/s**2"),
)


PackagingCellQualifiedBrake = PackagingCellRestrictedMode.variant(
    name="The restricted mode with the brake qualified at a lower bound",
    description="Load testing qualifies a weaker brake; the ceiling and the restricted stop against it.",
    deceleration=Q_(12.0, "m/s**2"),  # scenario assumption: lower qualified braking bound
)


ShieldCoverageBill = _st.ShieldCoverageBillScenario(
    name="A shielded hazard claim against a borrowed 1e-9 per hour target",
    description="Zero hazardous proposals in one release cycle of a 100-unit fleet, a shield of 0.9999 "
    "class coverage, common cause set to zero; the evidence the claim still lacks.",
    fleet_units=100,  # scenario assumption: fleet size
    cycle_days=Q_(90.0, "day"),  # scenario assumption: release cycle
    hours_per_day=Q_(24.0, "hour/day"),  # scenario assumption: continuous operation
    coverage=0.9999,  # scenario assumption: class-specific interception coverage
    common_cause_rate=Q_(0.0, "1/hour"),  # scenario assumption: hardware and common cause set to zero
    target_rate=Q_(1.0e-9, "1/hour"),  # scenario assumption: borrowed civil-aviation band, not statutory
    confidence=0.95,  # scenario assumption: chosen one-sided level
)


WarehouseLatchSeriesSpring = _st.LatchSeriesSpringScenario(
    name="A series spring at the door latch under the declared force floor",
    description="Spring sized for the fast approach; coasting peaks with torque cut at the tripwire.",
    site=_Embodied.Scenario.WarehouseAisle,
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
)
