"""Named agents scenarios for coordination.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Coordinating several agents (added 2026-10-20) -------------------------
RepairPartitionScreen = _st.PartitionScreenScenario(
    name="A repair partition with 80 percent divisible work and 0.02 overhead per added worker",
    description="Elapsed time over the serial reference for one to sixteen workers.",
    parallel_fraction=0.80,  # scenario assumption
    overhead_per_extra_worker=0.02,  # scenario assumption: normalized by the serial time
    max_workers=16,  # scenario assumption: resource ceiling
)


RepairPartitionScreenHalvedOverhead = RepairPartitionScreen.variant(
    name="The same partition with the per-worker overhead halved",
    overhead_per_extra_worker=RepairPartitionScreen.overhead_per_extra_worker / 2,
)


RepairPartitionScalability = _st.ScalabilityLawScenario(
    name="The repair partition as a universal scalability law: sigma 0.20, kappa 0.02",
    description="sigma = 1 - p and kappa = c reproduce the screen's elapsed times as capacities.",
    contention=1 - RepairPartitionScreen.parallel_fraction,
    coherency=RepairPartitionScreen.overhead_per_extra_worker,
    search_limit=64,  # scenario assumption: counts searched
)


ResearchFanoutScalability = RepairPartitionScalability.variant(
    name="The orchestrator-worker research agent: sigma 0.10, kappa 0.002",
    description="Independent search paths reconcile rarely, so coherency is a tenth of the repair partition's.",
    contention=_Agents.Research.OrchestratorWorker.serial_fraction,
    coherency=_Agents.Research.OrchestratorWorker.coherency_coefficient,
)


ResearchSystemTokenBreakEven = _st.DelegationTokenBreakEvenScenario(
    name="Delegation at the research system's 15x against 4x token multiples over chat",
    description="Break-even acceptance at a 3.75x token multiple, and the reported 90.2 percent gain "
    "read as an acceptance multiple; a single agent that already accepts 60 percent.",
    agent_tokens_vs_chat=_Agents.Research.OrchestratorWorker.agent_token_multiple,
    multi_agent_tokens_vs_chat=_Agents.Research.OrchestratorWorker.fanout_token_multiple,
    reported_gain=90.2 / 100,  # Hadfield et al. (2025), How we built our multi-agent research system...
    single_acceptance=0.60,  # scenario assumption: a task one agent usually completes
)


InterfaceMigrationSchedule = _st.DependencyScheduleScenario(
    name="A seven-node interface migration on four slots",
    description="Specification, three module changes, an integration step, a test step, and the "
    "final check; v2 and v3 tie on the critical path.",
    durations={
        1: Q_(20.5, "s"),
        2: Q_(42, "s"),
        3: Q_(42, "s"),
        4: Q_(42, "s"),
        5: Q_(43, "s"),
        6: Q_(16.5, "s"),
        7: Q_(69, "s"),
    },  # scenario assumption: node durations
    predecessors={1: [], 2: [1], 3: [1], 4: [1], 5: [2, 3], 6: [4], 7: [5, 6]},
    slots=4,  # scenario assumption
)


SixChildReferencePassing = _st.ReferencePassingScenario(
    name="Six children: a pasted 48,000-token slice or a 350-token reference and 3,600 tokens of reads",
    description="Each supplied input token is charged once at a hypothetical $3 per million.",
    workers=6,  # scenario assumption
    slice_tokens=48_000,  # scenario assumption
    reference_tokens=350,  # scenario assumption
    read_tokens=3_600,  # scenario assumption
    input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical input price
)


ThousandFileSharedWrites = _st.SharedWriteScenario(
    name="Workers each writing 10 of 1,000 files in one shared workspace",
    description="Uniform, independent write sets; the exact disjoint product and its birthday approximation.",
    files=1_000,  # scenario assumption
    writes_per_worker=10,  # scenario assumption
    max_fleet=16,  # scenario assumption: fleet sizes evaluated
    small_fleet=2,  # scenario assumption
    large_fleet=8,  # scenario assumption
)


FiveCorrelatedReviewers = _st.CorrelatedReviewersScenario(
    name="Five reviewers, each falsely approving 20 percent, with mixture correlation 0.40",
    description="Majority and unanimous false approval, independent against beta-binomial, and the "
    "floor under the majority as the panel grows to fifteen.",
    reviewers=5,  # scenario assumption
    false_approval=0.20,  # scenario assumption: marginal false-approval rate
    correlation=0.40,  # scenario assumption: pairwise correlation of the propensity
    large_ensemble=15,  # scenario assumption
)


EightWorkerDeadlineFanout = _st.FanoutDeadlineScenario(
    name="Eight workers, each on time 90 percent of the time, against a 10 s deadline",
    description="All eight results against any six of eight interchangeable ones.",
    workers=8,  # scenario assumption
    required=6,  # scenario assumption: the contract's quorum
    on_time_probability=0.90,  # scenario assumption
    deadline=Q_(10, "s"),  # scenario assumption
    max_fleet=16,  # scenario assumption: fleet sizes tabulated
)


DiffReviewCreditWindow = _st.CreditWindowScenario(
    name="A diff producer at 5 items/s feeding a reviewer of 500-token critiques at 40 tokens/s",
    description="Ten minutes unbounded against a window of eight credits; the window for a one-minute residence.",
    producer_rate=Q_(5, "1/s"),  # scenario assumption
    decode_rate=Q_(40, "token/s"),  # scenario assumption
    tokens_per_item=Q_(500, "token"),  # scenario assumption
    horizon=Q_(600, "s"),  # scenario assumption
    credits=8,  # scenario assumption
    residence_target=Q_(60, "s"),  # scenario assumption
)


MatchedDelegationCohort = _st.DelegationCohortScenario(
    name="100 migrations, single controller against delegation, under a $20 ceiling",
    description="Both accept 80 by the 600 s cutoff; only delegation meets 75 by the 300 s deadline.",
    attempted=100,  # scenario assumption
    accepted_floor=80,  # scenario assumption
    on_time_floor=75,  # scenario assumption
    deadline=Q_(300, "s"),  # scenario assumption
    cutoff=Q_(600, "s"),  # scenario assumption
    budget=Q_(20, "USD"),  # scenario assumption
    input_price=SixChildReferencePassing.input_price,
    output_price=Q_(10.00, "USD / Mtoken"),  # scenario assumption: hypothetical output price
    service_price=Q_(0.01, "USD / s"),  # scenario assumption: metered tool and assessment time
    single=_st.CohortArmRecord(
        accepted=80,
        on_time=70,
        median_accepted_time=Q_(240, "s"),  # scenario assumption
        input_tokens=Q_(2_000_000, "token"),
        output_tokens=Q_(400_000, "token"),  # scenario assumption
        service_time=Q_(200, "s"),  # scenario assumption
        discard_input_tokens=Q_(500_000, "token"),
        discard_output_tokens=Q_(100_000, "token"),  # scenario assumption
        discard_service_time=Q_(50, "s"),  # scenario assumption
    ),
    delegated=_st.CohortArmRecord(
        accepted=80,
        on_time=78,
        median_accepted_time=Q_(180, "s"),  # scenario assumption
        input_tokens=Q_(3_000_000, "token"),
        output_tokens=Q_(600_000, "token"),  # scenario assumption
        service_time=Q_(300, "s"),  # scenario assumption
        discard_input_tokens=Q_(1_000_000, "token"),
        discard_output_tokens=Q_(200_000, "token"),  # scenario assumption
        discard_service_time=Q_(100, "s"),  # scenario assumption
    ),
)
