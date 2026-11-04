"""Named robotics scenarios for oversight.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..robotics.supervision import WarehouseOutOfLoopTakeoverAtAisleSpeed


# --- Responsible physical AI: decisions, oversight, stages, groups, records (added 2026-10-20) ---
AdmittedCommandDecisionChain = _st.DecisionChainScenario(
    name="Six separately kept decisions behind one admitted aisle command",
    description="Chain completeness p^6 at three per-link retrieval probabilities.",
    links=6,  # scenario assumption: policy, enforcer, authority, envelope, crossing rule, staffing
    link_probabilities=(0.99, 0.95, 0.90),  # scenario assumption: illustrative retrieval probabilities
)


WarehouseRackEndOversightTime = _st.OversightTimeScenario(
    name="The out-of-loop takeover against the time a person stepping out at the rack end leaves",
    description="Clear distance at the aisle speed, the design out-of-loop takeover, and its travel.",
    budget=WarehouseOutOfLoopTakeoverAtAisleSpeed,
)


WarehouseRemoteAssistantQueue = _st.SupervisorQueueScenario(
    name="One remote assistant serving twenty warehouse machines",
    description="M/M/1 utilization, mean wait, and 99th-percentile wait; the fleet a one-half cap allows.",
    request_rate=Q_(0.5, "1/hour"),  # scenario assumption: illustrative help requests per machine-hour
    service_time=Q_(3.0, "minute"),  # scenario assumption: illustrative mean time per request
    machines=20,  # scenario assumption: illustrative fleet
    utilization_cap=0.5,  # scenario assumption: chosen staffing ceiling
)


WarehouseRemoteAssistantQueueTenMachines = WarehouseRemoteAssistantQueue.variant(
    name="One remote assistant serving ten warehouse machines",
    description="The same requests from half the fleet.",
    machines=10,  # scenario assumption: illustrative fleet
)


WarehouseRemoteAssistantQueueThirtyTwoMachines = WarehouseRemoteAssistantQueue.variant(
    name="One remote assistant serving thirty-two warehouse machines",
    description="The same requests from a larger fleet.",
    machines=32,  # scenario assumption: illustrative fleet
)


WarehouseRemoteAssistantLabor = _st.SupervisionLaborScenario(
    name="Person-hours a twenty-machine warehouse fleet moves into rescue, watching, and sweeps",
    description="An assistant staffed through two shifts at the queue's utilization, plus corridor sweeps.",
    queue=WarehouseRemoteAssistantQueue,
    operating_day=Q_(16.0, "hour"),  # scenario assumption: two shifts of operation
    sweeps=4,  # scenario assumption: inspections of the restricted corridors
    sweep_time=Q_(20.0, "minute"),  # scenario assumption: one corridor sweep
    shift=Q_(8.0, "hour"),  # scenario assumption: one full-time shift
)


WarehouseSupervisedStageExit = _st.HazardExposureScenario(
    name="Failure-free hours ten supervised warehouse machines need to leave their stage",
    description="Zero-failure exposure bounding one safety-relevant intervention per 1,000 h at 95 percent.",
    rates=[Q_(1.0e-3, "1/hour")],  # scenario assumption: chosen target rate
    confidence=0.95,  # scenario assumption: chosen, one-sided
    rigs=10,  # scenario assumption: machines in the supervised stage
    hours_per_day=Q_(16.0, "hour/day"),  # scenario assumption: two shifts
)


WarehouseGroupDetectionShortfall = _st.DetectionShortfallScenario(
    name="A stop sized for the best-detected group, met by three groups seen less often",
    description="Frames to a reliable detection, distance lost, impact speed and energy at the aisle speed.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    all_miss_probability=1.0e-3,  # scenario assumption: chosen all-frames miss probability
    reference=0.95,  # scenario assumption: illustrative best-detected group
    groups=(0.85, 0.75, 0.60),  # scenario assumption: illustrative lower-detection groups
)


PedestrianGroupEvaluationSample = _st.SubgroupSampleScenario(
    name="A 10,000-person detection test set with a 2 percent group",
    description="Exact 95 percent intervals per group and the per-group sample that tells 0.95 from 0.92.",
    total=10_000,  # scenario assumption: illustrative test set
    small_share=0.02,  # scenario assumption: illustrative group share
    recall=0.95,  # scenario assumption: illustrative detection rate in both groups
    rate_a=0.95,  # scenario assumption: chosen gap to resolve
    rate_b=0.92,  # scenario assumption: chosen gap to resolve
    significance=0.05,  # scenario assumption: chosen two-sided level
    power=0.80,  # scenario assumption: chosen
)


WarehouseCrossingExposure = _st.ExposureConcentrationScenario(
    name="Machine-person encounters concentrated at three warehouse crossings",
    description="Per-person exposure of the crossing users and of everyone else against the site average.",
    encounter_share=0.80,  # scenario assumption: illustrative share of encounters
    user_share=0.10,  # scenario assumption: illustrative share of workers
)


WarehouseEventTriggeredRetention = _st.EventRetentionScenario(
    name="Event-triggered camera retention on the warehouse machine",
    description="The pre-trigger window plus a post-event capture per event, against keeping the whole day.",
    pre_trigger=_Embodied.Scenario.WarehouseAisle.pre_trigger_window,
    post_trigger=Q_(25.0, "s"),  # scenario assumption: chosen capture after the event
    events_per_day=20,  # scenario assumption: illustrative events per machine-day
    operating_day=Q_(16.0, "hour"),  # scenario assumption: two shifts
)


CruiseDisclosurePenalties = _st.DisclosurePenaltyScenario(
    name="Penalties Cruise paid for its account of the October 2, 2023 collision",
    description="CPUC days at the daily maximum, the NHTSA consent-order penalty, and the DOJ fine.",
    days_withheld=15,  # California Public Utilities Commission, Decision 24-07-004, Presidi...
    daily_maximum=7500,  # California Public Utilities Commission, Decision 24-07-004, Presidi...
    others={
        "NHTSA": 1500000,  # NHTSA, 'NHTSA Announces Consent Order with Cruise After Company Fai...
        "DOJ": 500000,  # U.S. Attorney's Office, Northern District of California, 'Cruise Ad...
    },
)
