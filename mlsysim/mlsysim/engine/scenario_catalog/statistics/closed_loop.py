"""Named statistics scenarios for closed loop.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..robotics.demonstrations import LatchDemonstrationCampaign, LatchScenarioTable


# --- Closed-loop evaluation: the latch campaign, sequential tests, exposure (added 2026-10-20) ---
LatchEvaluationCampaign = _st.ClosedLoopCampaignScenario(
    name="Twenty clean closed-loop runs of the latch policy at the cage door, an illustrative campaign",
    description="The nominal-plate cell at the guarded approach, base stopped; the zero-failure bound, "
    "the runs and cell time a 99.9 percent target needs, and the contrasts it is taught with.",
    site=_Embodied.Scenario.WarehouseAisle,
    table=LatchScenarioTable,
    standoff=Q_(2.0, "m"),  # scenario assumption: base start distance from the door
    run=LatchDemonstrationCampaign.attempt,  # drive-in, stop, guarded unlatch and swing
    reset=LatchDemonstrationCampaign.reset,  # re-latch, return to the standoff, telemetry flush
    jam_reset=Q_(60.0, "s"),  # scenario assumption: clear a jammed pin, recalibrate, return
    operators=2,  # scenario assumption: safety operators watching each run
    shift=Q_(8.0, "hour"),  # scenario assumption: one operator shift
    clean_runs=20,  # scenario assumption: the campaign's clean runs
    confidence=0.95,  # scenario assumption: one-sided confidence
    target_rate=1e-3,  # scenario assumption: the 99.9 percent target
    estimate_target_rate=0.005,  # scenario assumption: the 99.5 percent budget estimate
    weak_policy_one_in=8,  # scenario assumption: a policy failing one run in eight
    recorded_failures=1,  # scenario assumption: one failure in the same campaign
    joint_cells=2,  # scenario assumption: nominal and displaced-plate cells
    looks=5,  # scenario assumption: independent 5 percent looks
    omissions=2,  # scenario assumption: soft penalty contact, missing transport lag
    fallacy_runs=1000,  # scenario assumption: the perfect-record fallacy
)


LatchSequentialTest = _st.SequentialTestScenario(
    name="Wald's sequential test for the latch campaign",
    description="Unacceptable 10 percent against acceptable 1 percent failure, alpha 0.05, beta 0.10; "
    "the score after 25 clean runs and one failure.",
    p_unacceptable=0.10,  # scenario assumption
    p_acceptable=0.01,  # scenario assumption
    alpha=0.05,  # scenario assumption: false acceptance under H0
    beta=0.10,  # scenario assumption: false rejection under H1
    trace_clean_runs=25,  # scenario assumption: clean runs before the first failure
)


LatchCheckpointComparison = _st.ArmComparisonScenario(
    name="Two latch checkpoints, 20 interleaved runs each: 20 of 20 against 16 of 20",
    description="Fisher's exact test, and the runs per arm that separate 0.90 from 0.80 at 80 percent power.",
    arm_runs=20,  # scenario assumption
    successes_a=20,  # scenario assumption
    successes_b=16,  # scenario assumption
    alpha=0.05,  # scenario assumption: two-sided
    power=0.80,  # scenario assumption
    p_high=0.90,  # scenario assumption
    p_low=0.80,  # scenario assumption
)


HazardExposureWall = _st.HazardExposureScenario(
    name="Failure-free hours to bound hazards of 1e-4, 1e-6, and 1e-9 per hour",
    description="The exposure wall at 95 percent one-sided confidence, per single machine, against one "
    "rig-year and a 100-rig fleet.",
    rates=[
        Q_(1e-4, "1/hour"),  # scenario assumption: an industrial site requirement
        Q_(1e-6, "1/hour"),  # scenario assumption: teaching target
        Q_(1e-9, "1/hour"),
    ],  # scenario assumption: aviation's catastrophic band
    confidence=0.95,  # scenario assumption
    rigs=100,  # scenario assumption: the fleet-testing fallacy
    rig_time=Q_(1.0, "year"),  # one rig running continuously for a year
)


WarehouseMissionExposure = _st.MissionExposureScenario(
    name="Failure-free retrieval missions that bound 1e-4 per hour on the warehouse manipulator",
    description="Missions of 360 s at the drive limit with 120 s resets; calendar time and wheel revolutions.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    hazard_rate=HazardExposureWall.rates[0],
    mission=Q_(360.0, "s"),  # scenario assumption: one retrieval mission
    reset=Q_(120.0, "s"),  # scenario assumption: dock recharge and telemetry reset
    confidence=0.95,  # scenario assumption
    relaxed_mtbf=Q_(1000.0, "hour"),  # scenario assumption: the relaxed target of the checkpoint
)


WarehouseSupervisedMissions = _st.InterventionCampaignScenario(
    name="38 interventions in 100 supervised retrieval missions",
    description="Active hours, distance, MTBI, intervention density, and the Garwood 95 percent rate interval.",
    mission=WarehouseMissionExposure,
    missions=100,  # scenario assumption
    interventions=38,  # scenario assumption: interventions of the declared class
    alpha=0.05,  # scenario assumption: two-sided interval
)


AisleHeadingDrift = _st.HeadingDriftScenario(
    name="A persistent 1 degree heading bias at the drive limit in the warehouse aisle",
    description="Open-loop angular error against where the closed loop spends the side clearance.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    site=_Embodied.Scenario.WarehouseAisle,
    heading_offset=Q_(1.0, "degree"),  # scenario assumption: navigation heading bias
)


StiffSurfaceResidual = _st.StiffContactScenario(
    name="A 2 mm residual against a hypothetical 5e5 N/m stiff surface",
    description="The force a millimeter-scale tracking residual commands, and the penetrations at the "
    "latch tripwire and limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    stiffness=Q_(5e5, "N/m"),  # scenario assumption: a stiff surface, not the running machine's latch
    residual=Q_(2.0, "mm"),  # scenario assumption: tracking residual
)


LatchForceTail = _st.ForceTailScenario(
    name="A benign mean latch force with a damaging 99th-percentile peak",
    description="100 door trials: mean under the tripwire, tail past the structural limit.",
    site=_Embodied.Scenario.WarehouseAisle,
    trials=100,  # scenario assumption
    mean_force=Q_(12.0, "N"),  # scenario assumption
    p99_force=Q_(115.0, "N"),  # scenario assumption: at an angular offset
)


RoadMilesToDemonstrate = _st.RoadExposureScenario(
    name="RAND's miles to demonstrate against Waymo's rider-only miles through January 2025",
    description="Kalra and Paddock 2016, Table 1, and Kusano et al. 2025 (millions of miles).",
    rates={
        "fatalities": 1.09,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "injuries": 77,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "crashes": 190,
    },  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
    zero_failure={
        "fatalities": 275,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "injuries": 3.9,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "crashes": 1.6,
    },  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
    precision={
        "fatalities": 8800,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "injuries": 125,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "crashes": 51,
    },  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
    improvement={
        "fatalities": 11000,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "injuries": 161,  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
        "crashes": 65,
    },  # Kalra, N. and Paddock, S. M. (2016), Driving to Safety: How Many Mi...
    fleet_miles=56.7,  # Kusano, K. D. et al. (2025), Comparison of Waymo Rider-Only Crash R...
    confidence=0.95,  # RAND row (1): 95 percent upper limit
)


AutosteerCrashRateDenominator = _st.CrashRateDenominatorScenario(
    name="NHTSA's Autosteer crash-rate drop against the complete-exposure cohort",
    description="Airbag-deployment crashes per million miles before and after installation (NHTSA PE 16-007; "
    "Quality Control Systems 2019).",
    reported_before=1.3,  # NHTSA Office of Defects Investigation, ODI Resume, Preliminary Eval...
    reported_after=0.8,  # NHTSA Office of Defects Investigation, ODI Resume, Preliminary Eval...
    cohort_before=0.76,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
    cohort_after=1.21,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
    exact_exposure_share=13 / 100,  # Quality Control Systems Corp. (Whitfield, R. A.), NHTSA's Implausib...
)
