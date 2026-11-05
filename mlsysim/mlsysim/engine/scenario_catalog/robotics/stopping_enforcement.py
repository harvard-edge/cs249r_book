"""Named robotics scenarios for stopping enforcement.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....physics.drivetrain import calc_traction_deceleration_limit as _traction_limit


# --- The warehouse base's finished stopping budget and its enforcer (added 2026-10-20) ---
# Site record Embodied.Scenario.WarehouseAisle and machine
# Embodied.MobileManipulator.WarehouseMobileManipulator: the elementary stop
# plus the resident S-curve (jerk bound j_brake), the tracking inset, and,
# in the walking-person cases, a person closing the gap. Ceilings are read
# from speed_ceiling; the speed field sets the budget rows.
WarehouseFinishedStopAtDriveLimit = _st.StoppingBudgetScenario(
    name="Warehouse base's finished stopping budget at its drive limit",
    description="Braking, the two allowances, brake onset, lease path, observation age, the "
    "S-curve's jerk ramps, and the tracking inset, against the rack-end clear distance.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
    jerk=_Embodied.Scenario.WarehouseAisle.j_brake,
    tracking_bound=_Embodied.Scenario.WarehouseAisle.eps_track,
)


WarehouseFinishedStopAtAisleSpeed = WarehouseFinishedStopAtDriveLimit.variant(
    name="Warehouse base's finished stopping budget at the candidate aisle speed",
    description="The same terms at the chosen aisle speed; the spare over the speed is the "
    "unallocated timing allowance.",
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
)


WarehouseFinishedStopWithDispatch = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Warehouse base's finished stopping budget with dispatch-to-admission charged",
    description="The pre-brake delay plus the dispatch-to-admission allocation: the complete "
    "static-obstacle ceiling, and the reserve left at the aisle speed.",
    delay=(_Embodied.Scenario.WarehouseAisle.tau_delay + _Embodied.Scenario.WarehouseAisle.t_dispatch_to_admission).to(
        "ms"
    ),
)


WarehouseFinishedStopInspectedFloor = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Warehouse base's finished stopping budget on the inspected low-traction floor",
    description="Deceleration capped by the inspected floor's friction, all weight on the driven "
    "wheels; the jerk bound belongs to the load and is unchanged.",
    deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_inspected_floor),
)


WarehouseFinishedStopInspectedFloorWithDispatch = WarehouseFinishedStopInspectedFloor.variant(
    name="Inspected-floor finished budget with dispatch-to-admission charged",
    description="The inspected-floor ceiling after the dispatch-to-admission allocation.",
    delay=WarehouseFinishedStopWithDispatch.delay,
)


WarehouseFinishedStopWalkingPerson = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Warehouse base's finished stopping budget against a person walking toward it",
    description="A person at the ISO 13855 approach speed closes the gap through the delay and the whole S-curve stop.",
    approach_speed=_Embodied.Scenario.WarehouseAisle.v_human_approach,
)


WarehouseFinishedStopWalkingPersonWithDispatch = WarehouseFinishedStopWalkingPerson.variant(
    name="Walking-person finished budget with dispatch-to-admission charged",
    description="The walking-person ceiling after the dispatch-to-admission allocation; the person "
    "keeps walking through that interval too.",
    delay=WarehouseFinishedStopWithDispatch.delay,
)


WarehouseFinishedStopTenMsSlower = WarehouseFinishedStopAtAisleSpeed.variant(
    name="Warehouse base's finished stopping budget with 10 ms more pre-brake delay",
    description="Latency sensitivity of the ceiling.",
    delay=(_Embodied.Scenario.WarehouseAisle.tau_delay + Q_(10.0, "ms")).to("ms"),  # scenario assumption
)


WarehouseHostLeaseStall = WarehouseFinishedStopAtAisleSpeed.variant(
    name="A 200 ms host stall in place of the 60 ms chunk lease at the aisle speed",
    description="A lease enforced by a non-real-time host keeps the last proposal for the stall; "
    "added_delay is the stall beyond the lease.",
    added_delay=(
        Q_(200.0, "ms")  # scenario assumption: proposer stall before any brake
        - _Embodied.Scenario.WarehouseAisle.t_lease
    ).to("ms"),
)


WarehouseIntegratorUnwind = WarehouseFinishedStopAtAisleSpeed.variant(
    name="A wound-up integrator that delays braking 150 ms at the aisle speed",
    description="An unprotected integrator must unwind before braking starts; added_delay is the unwind.",
    added_delay=Q_(150.0, "ms"),  # scenario assumption: illustrative unwind time
)


LoadedArmJointGravityStop = _st.GravityBiasedJointStopScenario(
    name="A loaded arm joint braking with and against gravity",
    description="The running machine's arm at its peak joint torque, a payload-carrying joint "
    "approaching at 1.2 rad/s with a 0.80 m lever, gravity holding 75 percent of the "
    "peak; the kinetic barrier priced at 110 and 70 mm of tool clearance; the "
    "protective separation against a person walking in.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    inertia=Q_(3.0, "kg*m**2"),  # scenario assumption: joint with payload (illustrative)
    lever=Q_(0.80, "m"),  # scenario assumption: illustrative
    joint_speed=Q_(1.2, "1/s"),  # scenario assumption: rad/s (illustrative)
    bias_fraction=0.75,  # scenario assumption: share of peak torque holding the load
    barrier_gain=Q_(10.0, "1/s"),  # scenario assumption: chosen barrier gain
    barrier_clearances=[Q_(110.0, "mm"), Q_(70.0, "mm")],  # scenario assumption: illustrative
    site=_Embodied.Scenario.WarehouseAisle,
)


TwoJointTorqueProjection = _st.TorqueHalfSpaceProjectionScenario(
    name="A two-joint torque proposal against one barrier half-space",
    description="Proposal (3.5, 3) N m against 2 tau_1 + tau_2 <= 5 N m: the least-change "
    "projection against a per-axis clamp of joint 1 to 2 N m (abstract illustration).",
    proposal=[3.5, 3.0],  # scenario assumption: N m
    normal=[2.0, 1.0],  # scenario assumption
    bound=5.0,  # scenario assumption: N m
    clamp=[2.0, 3.0],  # scenario assumption: per-axis clamp, N m
    actuator_limit=5.0,  # scenario assumption: symmetric per-joint limit, N m
)


WarehouseEnforcerCycleBudget = _st.CycleBudgetScenario(
    name="Stage budget of the warehouse machine's 1 kHz permission check",
    description="Declared sub-budgets that sum to the site's declared WCET inside its deadline.",
    site=_Embodied.Scenario.WarehouseAisle,
    stages=[  # scenario assumption: chosen sub-budgets
        ("state acquisition", Q_(30.0, "us")),
        ("proposal and lease check", Q_(15.0, "us")),
        ("tracking check", Q_(10.0, "us")),
        ("QP solve", Q_(150.0, "us")),
        ("output write", Q_(15.0, "us")),
        ("reserve", Q_(30.0, "us")),
    ],
)


WarehouseBaseHoldDrift = _st.HoldIntervalDriftScenario(
    name="Drift of the warehouse base over one held 1 ms command at its drive limit",
    description="Zero-order hold: travel at the drive limit plus an independent approach-"
    "acceleration upper bound over one permission tick.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    acceleration=Q_(2.0, "m/s**2"),  # scenario assumption: approach-acceleration upper bound
    period=_Embodied.Scenario.WarehouseAisle.t_tick,
)
