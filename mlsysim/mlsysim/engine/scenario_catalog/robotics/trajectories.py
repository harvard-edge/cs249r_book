"""Named robotics scenarios for trajectories.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....actuators.registry import Actuators as _Actuators
from ..robotics.grounding import ConveyorMugIntentLeaseExpiry


# --- Trajectory planning on the warehouse mobile manipulator (added 2026-10-20) ---
# Joint rates through the arm's Jacobian, a replanner's mission exposure, chunk
# replacement, the jerk-limited stops of the arm and the base, a trajectory
# record's deadlines, a seam bridge on the Panda-class joint, the loaded
# shoulder's swing, the base's dynamic window, and a stale anchor.
WarehouseArmRightAngleElbow = _st.TwoLinkJointRateScenario(
    name="Warehouse arm at a right-angle elbow, TCP speed limit in the weakest direction",
    description="Planar two-link abstraction of the arm (FER link lengths): joint rates the 1 m/s TCP "
    "limit asks, exact and with 50 mm damped least squares.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    elbow_angle=Q_(90.0, "degree"),  # scenario assumption: right-angle elbow
    damping=Q_(50.0, "mm"),  # scenario assumption: illustrative damping, not a vendor setting
)


WarehouseArmNearStraightElbow = WarehouseArmRightAngleElbow.variant(
    name="Warehouse arm at a nearly straight elbow, TCP speed limit in the weakest direction",
    description="The right-angle case with the elbow straightened to 10 degrees, near the singular pose.",
    elbow_angle=Q_(10.0, "degree"),  # scenario assumption: nearly straight elbow
)


GenericReplannerMissionExposure = _st.HandoffExposureScenario(
    name="A p99 replanner at 5 Hz over a ten-minute mission",
    description="Handoffs, expected late handoffs, the chance all are on time, and the per-handoff "
    "budget a 1-in-1,000 mission target leaves (independent misses).",
    mission_time=Q_(600.0, "s"),  # scenario assumption: ten-minute mission
    replan_rate=Q_(5.0, "Hz"),  # scenario assumption: a generic replanner, slower than the chunk renewal
    late_probability=0.01,  # definitional: exceedance of a p99 budget
    mission_budget=1e-3,  # scenario assumption: one failed mission in 1,000
)


WarehouseArmChunkReplacement = _st.ChunkReplacementScenario(
    name="Replacing the warehouse arm's admitted chunk",
    description="Capture-to-admission latency from the observation age and the dispatch-to-admission "
    "allocation, against the 50 ms renewal and the 60 ms chunk lease.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    site=_Embodied.Scenario.WarehouseAisle,
    interference_tail=Q_(150.0, "ms"),  # scenario assumption: inference pass under memory contention or preemption
)


WarehouseArmStallStop = _st.JerkLimitedStopScenario(
    name="Warehouse arm stalled at its TCP limit: lease path, then a jerk-limited stop",
    description="The proposer stalls; the chunk lease lapses, one tick, one bus cycle, and the brake "
    "onset pass, then an S-curve stop with an empty gripper, against the clearance ahead.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.tcp_speed_limit,
    deceleration=Q_(5.0, "m/s**2"),  # scenario assumption: arm stop deceleration bound, empty gripper
    delay=(
        _Embodied.Scenario.WarehouseAisle.t_lease
        + _Embodied.Scenario.WarehouseAisle.t_tick
        + _Embodied.Scenario.WarehouseAisle.t_bus
        + _Embodied.Scenario.WarehouseAisle.t_brake_onset
    ).to("ms"),  # brake onset charged as the arm's
    clear_distance=ConveyorMugIntentLeaseExpiry.clearance,
)


WarehouseArmOpenMugStop = _st.SpillLimitedStopScenario(
    name="Warehouse arm stalled while carrying an open mug",
    description="The stall stop with the deceleration capped where the liquid reaches the rim, its "
    "jerk rescaled with it, and the carry speed whose stop fits the clearance.",
    free_stop=WarehouseArmStallStop,
    inner_radius=Q_(40.0, "mm"),  # scenario assumption: mug inner radius
    freeboard=Q_(10.0, "mm"),  # scenario assumption: liquid surface below the rim
)


WarehouseBaseJerkLimitedStop = _st.JerkLimitedStopScenario(
    name="Warehouse base at its drive limit stopping along an S-curve at the rack end",
    description="The elementary budget's delay, localization bound, and margin with the constant-"
    "deceleration stop replaced by an S-curve whose jerk bound is the reference C2 "
    "polynomial's peak at the drive limit.",
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    jerk=_Embodied.Scenario.WarehouseAisle.j_brake,  # the site's stop jerk bound, shared with WarehouseFinishedStopAtDriveLimit
    delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
    site=_Embodied.Scenario.WarehouseAisle,
)


WarehouseArmCommitmentSchedule = _st.TrajectoryDeadlinesScenario(
    name="Deadlines of a longer-lease trajectory record for the warehouse arm",
    description="Preferred admission, full reserve, admission cutoff, commitment, and rest for a "
    "replacement requested 60 ms into the schedule, with the stall stop as its suffix.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    site=_Embodied.Scenario.WarehouseAisle,
    stop=WarehouseArmStallStop,
    request_delay=Q_(60.0, "ms"),  # scenario assumption: request delay
    latency_target=Q_(160.0, "ms"),  # scenario assumption: design-tail replacement-latency target
    schedule_slack=Q_(20.0, "ms"),  # scenario assumption: scheduling slack before preferred admission
    blend=Q_(240.0, "ms"),  # scenario assumption: blend deadline
    commit=Q_(300.0, "ms"),  # scenario assumption: physical suffix onset
    late_wait=Q_(200.0, "ms"),  # scenario assumption: tracking authority run past commitment
)


WarehouseArmShoulderSwing = _st.ShoulderSwingScenario(
    name="Warehouse arm shoulder swinging the rated payload at full reach",
    description="Payload-only holding torque and inertia with the Panda-class joint's reflected rotor "
    "inertia, and a 0.5 rad rest-to-rest swing, bang-bang and jerk-limited.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
    swing=Q_(0.5, "radian"),  # scenario assumption: swing amplitude
)


PandaClassJointSeamBridge = _st.SeamBridgeScenario(
    name="A 0.2 rad/s seam gap on the Panda-class joint: one tick against a 50 ms quintic bridge",
    description="Reflected rotor inertia alone (rigid), screened against the arm's torque-rate limit; "
    "the loaded step adds the rated payload at full reach.",
    joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    velocity_gap=Q_(0.2, "rad/s"),  # scenario assumption: reference velocity gap at the seam
    tick=_Embodied.Scenario.WarehouseAisle.t_tick,
    bridge_duration=Q_(50.0, "ms"),  # scenario assumption: bridge duration
    loaded_inertia=WarehouseArmShoulderSwing.result.effective_inertia,
)


PandaClassJointSeamBridgeHalved = PandaClassJointSeamBridge.variant(
    name="The Panda-class seam bridge at half its duration",
    description="The same gap bridged in half the time: torque doubles, jerk and torque rate quadruple.",
    bridge_duration=PandaClassJointSeamBridge.bridge_duration / 2,  # scenario assumption: halved bridge
)


PandaClassJointSeamBridgeCompressed = PandaClassJointSeamBridge.variant(
    name="The Panda-class seam bridge squeezed to 10 ms",
    description="A late admission leaves 10 ms for the same gap.",
    bridge_duration=Q_(10.0, "ms"),  # scenario assumption: compressed bridge
)


WarehouseBaseDynamicWindow = _st.DynamicWindowScenario(
    name="Warehouse base's dynamic window over one chunk renewal",
    description="Speeds reachable in 50 ms and the classical constant-deceleration admissible speed "
    "on the rack-end clear distance.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    site=_Embodied.Scenario.WarehouseAisle,
)


WarehouseArmStaleAnchor = _st.StaleAnchorScenario(
    name="A plan anchored at the pose sampled when the replacement was requested",
    description="A joint at 0.50 rad/s during an 80 ms replacement, at the arm's planar reach.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    joint_speed=Q_(0.50, "rad/s"),  # scenario assumption: constant joint speed
    delay=Q_(80.0, "ms"),  # scenario assumption: compute and transport of the replacement
)
