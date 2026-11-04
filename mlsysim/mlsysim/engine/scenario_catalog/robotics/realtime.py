"""Named robotics scenarios for realtime.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st


# --- The real-time permission path of the warehouse mobile manipulator (added 2026-10-20) ---
# The machine and its site record are Embodied.MobileManipulator.WarehouseMobileManipulator
# and Embodied.Scenario.WarehouseAisle (tick, bus cycle, lease, brake onset, the
# permission check's C, B, I, D, and the event intervals). Inputs neither record
# carries are stated here.
WarehouseJointLoopThroughTick = _st.PhaseLagBudgetScenario(
    name="A 20 Hz joint loop closed through the 1 kHz permission tick and the fieldbus",
    description="Sample-and-hold, the declared permission-check WCET, and one bus cycle, "
    "priced in phase at crossover against a 5 degree erosion allowance.",
    crossover=Q_(20.0, "Hz"),  # scenario assumption: illustrative joint-loop crossover
    sample_period=_Embodied.Scenario.WarehouseAisle.t_tick,
    compute_delay=_Embodied.Scenario.WarehouseAisle.enforcer_wcet,
    transport_delay=_Embodied.Scenario.WarehouseAisle.t_bus,
    phase_allowance=Q_(5.0, "degree"),  # scenario assumption: chosen phase erosion allowance
)


WarehouseJointLoopNextTickOutput = WarehouseJointLoopThroughTick.variant(
    name="The same joint loop with its output released at the next tick",
    description="The compute delay grows from the declared WCET to one whole tick.",
    compute_delay=_Embodied.Scenario.WarehouseAisle.t_tick,
)


WarehouseJointLoopSlowTick = WarehouseJointLoopThroughTick.variant(
    name="The same joint loop sampled by a 500 Hz tick",
    description="A 2 ms sample period, for the sample-and-hold term alone.",
    sample_period=Q_(2.0, "ms"),  # scenario assumption: 500 Hz comparison tick
)


WarehousePermissionTickResponse = _st.ResponseTimeScenario(
    name="The warehouse machine's permission check: response-time bound against its deadline",
    description="Declared WCET, blocking, and interference of the 1 kHz permission tick "
    "against its 400 microsecond deadline, all from the site record.",
    site=_Embodied.Scenario.WarehouseAisle,
)


WarehousePermissionTickLongBlocking = WarehousePermissionTickResponse.variant(
    name="The same permission check behind a longer tick-masking section",
    description="A new link driver masks the tick for 120 microseconds instead of 50.",
    blocking=Q_(120.0, "us"),  # scenario assumption: a longer tick-masking section
)


WarehouseServoCanFdOneBus = _st.CanFdBusScenario(
    name="All nine servo axes of the warehouse machine on one CAN FD bus",
    description="An 8-byte command and an 8-byte feedback frame per axis per 1 ms cycle, "
    "1 Mbps nominal and 5 Mbps data phase, worst-case stuffing.",
    num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes,
    payload_bytes=8,  # scenario assumption: position-mode record, each direction
    frames_per_axis=2,  # scenario assumption: one command and one feedback frame per axis
    baud_arbitration=Q_(1.0, "megabit/second"),  # scenario assumption: nominal rate
    baud_data=Q_(5.0, "megabit/second"),  # scenario assumption: data-phase rate
    cycle=_Embodied.MobileManipulator.WarehouseMobileManipulator.fieldbus_cycle,
)


WarehouseServoCanFdArmBus = WarehouseServoCanFdOneBus.variant(
    name="The warehouse machine's seven arm axes on their own CAN FD bus",
    description="One bus per limb: the arm joints only.",
    num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.arm.dofs,
)


WarehouseServoCanFdBaseBus = WarehouseServoCanFdOneBus.variant(
    name="The warehouse machine's two drive wheels on their own CAN FD bus",
    description="One bus per limb: the drive-wheel axes only.",
    num_axes=(
        _Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes
        - _Embodied.MobileManipulator.WarehouseMobileManipulator.arm.dofs
    ),
)


WarehouseServoEtherCat = _st.EtherCatCycleScenario(
    name="All nine servo axes of the warehouse machine on one EtherCAT chain",
    description="One summation frame carries every axis's 8-byte command and 8-byte feedback; "
    "each drive forwards it with a fixed hop delay.",
    num_axes=_Embodied.MobileManipulator.WarehouseMobileManipulator.servo_axes,
    bytes_per_axis=2 * 8,  # scenario assumption: command and feedback records share the frame
    baud_rate=Q_(100.0, "megabit/second"),  # 100BASE-TX link rate
    header_bytes=64,  # scenario assumption: preamble, Ethernet and EtherCAT headers, FCS
    hop_delay=Q_(0.6, "us"),  # scenario assumption: per-node forwarding delay
    cycle=_Embodied.MobileManipulator.WarehouseMobileManipulator.fieldbus_cycle,
)


WarehouseChunkLease = _st.LeaseBudgetScenario(
    name="The warehouse machine's 60 ms chunk lease at its drive limit",
    description="The lease path priced in the stopping budget, and the lease rule's floor "
    "(renewal period plus jitter) and ceilings (pre-brake ceiling less tick, bus, "
    "onset, and the evidence's age).",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    renewal_jitter=Q_(5.0, "ms"),  # scenario assumption: allowance for renewal variation
)


WarehouseChunkLeaseLengthened = WarehouseChunkLease.variant(
    name="The same chunk lease lengthened to 100 ms",
    description="A longer lease to ride out proposer hiccups, priced in clear distance.",
    lease=Q_(100.0, "ms"),  # scenario assumption: a lengthened lease
)
