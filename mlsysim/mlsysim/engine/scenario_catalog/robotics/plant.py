"""Named robotics scenarios for plant.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....actuators.registry import Actuators as _Actuators
from ....physics.constants import INSULATION_CLASS_F_LIMIT as _CLASS_F


# --- Physical-plant budgets of the warehouse mobile manipulator (added 2026-10-20) ---
# The machine and its site record are Embodied.MobileManipulator.WarehouseMobileManipulator
# and Embodied.Scenario.WarehouseAisle; the arm joint is the Panda-class stand-in
# Actuators.PandaClassJoint.CSG_25_50_ILM70x18. Inputs neither record carries are
# stated here.
WarehouseBaseStopBodyTerms = _st.StoppingBudgetScenario(
    name="Loaded warehouse base stopping from its drive limit, body terms only",
    description="Braking, the two spatial allowances, and travel during the brake onset; "
    "the delays the body does not own are left out.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    delay=_Embodied.Scenario.WarehouseAisle.t_brake_onset,
    added_delay=Q_(100, "millisecond"),  # scenario assumption: a delay step priced at the drive limit
)


WarehouseBaseStopFullDelay = _st.StoppingBudgetScenario(
    name="Loaded warehouse base stopping from its drive limit after the full pre-brake delay",
    description="The site's full pre-brake delay (brake onset, lease, tick, bus cycle, observation age).",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
    delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
)


WarehouseBaseStopFastSetpoint = _st.StoppingBudgetScenario(
    name="Loaded warehouse base stopping from a setpoint above its drive limit",
    description="The full pre-brake delay at 2 m/s, a speed-scaling case above the 1.5 m/s drive limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=Q_(2.0, "m/s"),  # scenario assumption: a proposed speed setpoint above the drive limit
    delay=_Embodied.Scenario.WarehouseAisle.tau_delay,
)


WarehouseBaseSpeedStep = _st.SpeedStepScenario(
    name="Raising the warehouse base's speed setpoint from 1.5 to 2 m/s",
    description="Braking travel added by the faster setpoint, and the delay that costs the same distance.",
    base=WarehouseBaseStopBodyTerms,
    raised=WarehouseBaseStopFastSetpoint,
)


WarehouseBaseBrakingCeilings = _st.BrakingCeilingsScenario(
    name="Ceilings on the loaded warehouse base's credible deceleration",
    description="Traction (dry and inspected floors), tip-over (center of gravity unmeasured), and the "
    "hub drives' torque at their current limit, with back-EMF headroom at the drive limit.",
    site=_Embodied.Scenario.WarehouseAisle,
)


WarehouseBaseHubDriveElectrical = _st.HubDriveElectricalScenario(
    name="Current rise and phase-short braking of the warehouse base's hub drives",
    description="Electrical time constant, the stall current ramp from a depleted pack, and the wheel "
    "speed at which shorted windings brake hardest.",
    site=_Embodied.Scenario.WarehouseAisle,
)


WristCameraGraspAgeProfile = _st.FreshnessProfileScenario(
    name="Wrist-camera grasp of a mug on the pick-station conveyor",
    description="An illustrative capture-to-onset age profile (P50, P99, max per stage) against the "
    "conveyor mug's freshness deadline; a measured profile of the integrated machine replaces it.",
    site=_Embodied.Scenario.WarehouseAisle,
    stages=[  # scenario assumption: illustrative stage profile, (P50, P99, max)
        ("Exposure and readout", Q_(11.2, "ms"), Q_(16.6, "ms"), Q_(16.7, "ms")),
        ("Transport and DMA", Q_(1.4, "ms"), Q_(3.2, "ms"), Q_(6.8, "ms")),
        ("Backbone and policy", Q_(18.5, "ms"), Q_(24.1, "ms"), Q_(38.4, "ms")),
        ("Scheduling and IPC", Q_(0.6, "ms"), Q_(2.1, "ms"), Q_(12.3, "ms")),
        ("Fieldbus", Q_(1.0, "ms"), Q_(1.0, "ms"), Q_(2.0, "ms")),
        ("Current rise", Q_(2.3, "ms"), Q_(3.0, "ms"), Q_(4.5, "ms")),
    ],
    focus_stage="Backbone and policy",
    system_p99=Q_(46.2, "ms"),  # scenario assumption: system-level profile
    system_p999=Q_(68.4, "ms"),  # scenario assumption: system-level tail
)


GearedRotorFlywheel = _st.GearedRotorScenario(
    name="A 50 g rotor through a 100:1 joint reduction",
    description="Reflected rotor inertia sized as a 10 cm solid flywheel, beside an illustrative link.",
    gear_ratio=100,  # scenario assumption: joint reduction
    rotor_mass=Q_(50.0, "gram"),  # scenario assumption
    rotor_inertia=Q_(5.6e-6, "kg * m**2"),  # scenario assumption: inertia of that rotor
    flywheel_radius=Q_(0.10, "meter"),  # scenario assumption: comparison flywheel radius
    load_inertia=Q_(8.5e-3, "kg * m**2"),  # scenario assumption: link inertia at the joint
)


PandaClassJointGearRatios = _st.GearRatioCostScenario(
    name="The Panda-class joint holding 87 N m at three gear ratios",
    description="TQ ILM 70x18 motor holding the Panda's maximum joint torque at the CSG-25-50 ratio, "
    "a catalog 100:1, and a quasi-direct-drive-class 10:1.",
    joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
    motor=_Actuators.FramelessMotor.TQ_ILM70x18,
    output_torque=_Embodied.Manipulator.Panda.max_torque,
    gear_efficiency=0.85,  # scenario assumption: the catalog's varies with load, speed, and temperature
    ratios=[
        int(_Actuators.PandaClassJoint.CSG_25_50_ILM70x18.gear_ratio),
        100,  # scenario assumption: a catalog CSG-25 ratio
        10,
    ],  # scenario assumption: quasi-direct-drive class ratio
)


PandaClassJointBurst = _st.ThermalBurstScenario(
    name="A threefold rated-current burst in the Panda-class joint",
    description="First-order lumped winding at fixed resistance, from ambient and from rated-current "
    "equilibrium, with the average-power duty screen and its periodic peak.",
    joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
    motor=_Actuators.FramelessMotor.TQ_ILM70x18,
    thermal_capacitance=Q_(120.0, "J/K"),  # scenario assumption: lumped; the data sheet publishes none
    thermal_resistance=Q_(2.5, "K/W"),  # scenario assumption: installed
    ambient=Q_(40.0, "degC"),  # scenario assumption: enclosed ambient
    limit=_CLASS_F,
    ceiling=Q_(140.0, "degC"),  # scenario assumption: working ceiling chosen below the Class F rating
    burst_factor=3,  # scenario assumption: below the motor's peak torque
    burst_on=Q_(3.0, "second"),  # scenario assumption
)


PandaClassElbowStall = _st.StallDeratingScenario(
    name="The Panda-class elbow holding 87 N m at stall",
    description="Steady winding temperature with copper's resistance feedback, and the hold the "
    "installation can sustain at the Class F limit.",
    joint=_Actuators.PandaClassJoint.CSG_25_50_ILM70x18,
    motor=_Actuators.FramelessMotor.TQ_ILM70x18,
    hold_torque=_Embodied.Manipulator.Panda.max_torque,
    gear_efficiency=PandaClassJointGearRatios.gear_efficiency,
    thermal_resistance=Q_(1.80, "K/W"),  # scenario assumption: unventilated limb
    ambient=Q_(40.0, "degC"),  # scenario assumption: limb ambient
    limit=_CLASS_F,
)


SealedJointCatalogDerating = _st.ContinuousDeratingScenario(
    name="A catalog continuous rating carried into a sealed 40 degC limb",
    description="Continuous torque at the working ceiling for a 25 degC catalog test against the "
    "enclosed ambient, and again with the installed thermal resistance doubled.",
    ceiling=PandaClassJointBurst.ceiling,
    catalog_ambient=Q_(25.0, "degC"),  # scenario assumption: catalog test ambient
    installed_ambient=PandaClassJointBurst.ambient,
    thermal_resistance_factor=2.0,  # scenario assumption: installed theta relative to the catalog heat sink
)


WarehouseDriveBusStop = _st.DriveBusScenario(
    name="The warehouse base's 48 V drive bus: coincident step and a stop from the drive limit",
    description="Sustained sag under the coincident inference and drive step, and the regenerative "
    "surge of a stop into a full pack, absorbed by the bus capacitance or a chopper.",
    site=_Embodied.Scenario.WarehouseAisle,
    link_capacitance=Q_(22000.0, "microfarad"),  # scenario assumption: bulk link capacitance
    regen_fraction=0.75,  # scenario assumption: share of kinetic energy returned to the bus
    candidate_resistor=Q_(3.9, "ohm"),  # scenario assumption: chosen chopper resistor
    candidate_power=Q_(1.0, "kilowatt"),  # scenario assumption: its pulse rating
)


CleanLoadedStopCampaign = _st.ZeroFailureScenario(
    name="3,000 failure-free loaded stops",
    description="The per-stop failure rate a clean campaign supports at 95 percent one-sided confidence.",
    trials=3000,  # scenario assumption
    confidence=0.95,
    target_rate=1e-4,  # scenario assumption: per-stop rate the record would need
)


ThrottledNavigationBackbone = _st.ThrottledStageScenario(
    name="The navigation backbone at two-thirds clock under thermal throttling",
    description="A compute-bound vision backbone stretched by DVFS, paid as blind travel at the drive limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    clock_ratio=2 / 3,  # scenario assumption: DVFS clock as a fraction of full clock
    speed=_Embodied.MobileManipulator.WarehouseMobileManipulator.base.max_velocity,
)
