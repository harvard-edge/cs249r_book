"""Named economics scenarios for cost accounting.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ..agents.context import RepairPrefixLayoutBill
from ..agents.deliberation import OutputCeilingThreeAllocations, ProbeVsGeneratedExplanation
from ..agents.harness import RepairTrajectoryBudgetMeter
from ..agents.model_calls import CodingAgentResend30Turns


# --- Agent cost accounting (added 2026-10-20) ------------------------------
# Types in scenario_types.py, solvers in engine/solvers/cost_accounting.py.
# Prices are hypothetical tariffs, not a provider schedule, unless a source
# is named; every other input is a stated teaching assumption.
FrontierModelRepairCohort = _st.CohortAttemptCostScenario(
    name="100 repair tickets on a frontier-priced model, one attempt each",
    description="Four turns of 16,000 input and 800 output tokens at $3 and $15 per million; $0.03 of "
    "environment per turn, $0.05 of assessment per attempt, 75 accepted; $10 of triage per "
    "rejected attempt.",
    tasks=100,  # scenario assumption
    accepted=75,  # scenario assumption
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=ProbeVsGeneratedExplanation.output_price,
    turns=4,  # scenario assumption
    input_tokens_per_turn=16_000,  # scenario assumption
    output_tokens_per_turn=800,  # scenario assumption
    environment_per_turn=Q_(0.03, "USD"),  # scenario assumption: tools, sandbox holding, cleanup
    assessment_per_attempt=Q_(0.05, "USD"),  # scenario assumption
    triage_per_rejected=Q_(10.00, "USD"),  # scenario assumption: engineer triage only
)


LowTokenPriceRepairCohort = FrontierModelRepairCohort.variant(
    name="The same 100 tickets on a model whose tokens cost a tenth as much",
    description="Seven turns of 28,000 input and 1,200 output tokens at $0.30 and $1.20 per million; "
    "25 accepted; the same environment, assessment, and triage charges.",
    accepted=25,  # scenario assumption
    input_price=Q_(0.30, "USD / Mtoken"),  # scenario assumption: hypothetical low tariff
    output_price=Q_(1.20, "USD / Mtoken"),  # scenario assumption: hypothetical low tariff
    turns=7,  # scenario assumption
    input_tokens_per_turn=28_000,  # scenario assumption
    output_tokens_per_turn=1_200,  # scenario assumption
)


RepairCohortTokenPriceComparison = _st.TokenPriceComparisonScenario(
    name="Frontier against low-token-price repair cohorts under one acceptance contract",
    description="Cost per accepted repair with and without triage, and the token, attempt, and acceptance cuts.",
    baseline=FrontierModelRepairCohort,
    alternative=LowTokenPriceRepairCohort,
)


RepairResendCacheBill = _st.ResendCacheBillScenario(
    name="The 30-turn resend trajectory billed fresh and with ideal prefix reuse",
    description="Hypothetical $3 fresh input, cached reads at 10 percent, $15 output per million "
    "tokens; a 10 percent miss on the eligible prefix for sensitivity.",
    trajectory=CodingAgentResend30Turns,
    input_price=RepairTrajectoryBudgetMeter.input_price,
    cached_fraction=RepairTrajectoryBudgetMeter.cached_fraction,
    output_price=RepairTrajectoryBudgetMeter.output_price,
    miss_fraction=0.10,  # scenario assumption: share of the eligible prefix that misses
)


RepairCompactionPayback = _st.CompactionPaybackScenario(
    name="Compacting the repair trajectory at call 16 to a 12,000-token summary",
    description="The summary costs 1,500 output tokens and is resent fresh; every later call reads "
    "18,500 fewer cached tokens.",
    bill=RepairResendCacheBill,
    compact_turn=16,  # scenario assumption
    compact_to=12_000,  # scenario assumption: compacted context, tokens
    summary_tokens=1_500,  # scenario assumption: summary output, tokens
)


RepairCompactionPaybackHalfCachedRate = RepairCompactionPayback.variant(
    name="The same compaction with cached reads at half the rate",
    description="Cached reads at 5 percent of the input price, as a newer published tariff charges.",
    bill=RepairResendCacheBill.variant(cached_fraction=RepairResendCacheBill.cached_fraction / 2),
)


CheapRetryAgainstStrongCall = _st.RetryPolicyScenario(
    name="A cheap attempt retried up to five times against one strong attempt",
    description="Cheap: passes 30 percent at $0.10; strong: 70 percent at $0.40; attempts independent.",
    pass_probability=0.30,  # scenario assumption
    attempt_cost=Q_(0.10, "USD"),  # scenario assumption: all-in
    max_attempts=5,  # scenario assumption
    strong_pass_probability=0.70,  # scenario assumption
    strong_attempt_cost=Q_(0.40, "USD"),  # scenario assumption
)


SerialRepairEpisode = _st.SerialEpisodeScenario(
    name="A short serial repair episode as disjoint intervals, model calls 4x faster",
    description="Six model calls, 245.60 s of required tests, sandbox creation and snapshot export, "
    "and runtime setup and status.",
    model_intervals=[Q_(x, "second") for x in (1.20, 8.40, 1.85, 11.20, 2.10, 4.50)],  # scenario assumption
    tests=Q_(245.60, "second"),  # scenario assumption
    lifecycle_intervals=[Q_(x, "second") for x in (14.80, 22.55)],  # scenario assumption
    runtime_intervals=[Q_(x, "second") for x in (0.12, 0.08)],  # scenario assumption
    model_speedup=4.0,  # scenario assumption
)


SerialRepairEpisodeSpeculation = _st.SpeculativeEpisodeScenario(
    name="Speculative decoding on the serial repair episode's model time",
    description="Five drafted tokens per round at 0.08 of a target step each; 3.6 or 0.3 kept per "
    "round. Treats the model time as pure decode, an optimistic bound.",
    episode=SerialRepairEpisode,
    draft_length=5,  # scenario assumption
    draft_cost_ratio=0.08,  # scenario assumption
    accepted_high=3.6,  # scenario assumption: drafts align
    accepted_low=0.3,  # scenario assumption: a hard reasoning task
)


CodeEditThreeTierCascade = _st.ModelCascadeScenario(
    name="A three-tier cascade over 100,000 code-editing tasks, one gate at every exit",
    description="Tier calls of $0.0004, $0.0020, and $0.0600 and 0.25, 0.80, and 4.50 s; a $0.0010, "
    "0.40 s gate; 60,000, 28,000, and 11,400 accepted; the frontier tier alone accepts "
    "97,000. A relaxed first-tier gate that falsely accepts 0.9 percent of its exits, at "
    "$10 of remediation each.",
    tasks=100_000,  # scenario assumption: completed cohort
    call_costs=[Q_(x, "USD") for x in (0.0004, 0.0020, 0.0600)],  # scenario assumption
    gate_cost=Q_(0.0010, "USD"),  # scenario assumption: same contract at every tier
    call_times=[Q_(x, "second") for x in (0.25, 0.80, 4.50)],  # scenario assumption
    gate_time=Q_(0.40, "second"),  # scenario assumption
    accepted_by_tier=[60_000, 28_000, 11_400],  # scenario assumption
    direct_accepted=97_000,  # scenario assumption: separate full-cohort baseline
    false_accept_rate=0.009,  # scenario assumption: a relaxed first-tier gate
    harm_per_false_accept=Q_(10.0, "USD"),  # scenario assumption: find and remediate one false accept
)


RepairTrajectoryQuotaAdmission = _st.TrajectoryAdmissionScenario(
    name="Repair trajectories admitted once a minute against a 2 million token-per-minute quota",
    description="20 minutes present, 60 model calls of 30,000 quota-counted tokens each; acceptance "
    "falling by a tenth of itself at fixed goodput.",
    arrival=Q_(1, "1 / minute"),  # scenario assumption
    residence=Q_(20, "minute"),  # scenario assumption
    calls=60,  # scenario assumption
    tokens_per_call=30_000,  # scenario assumption: every token the quota counts
    quota=Q_(2_000_000, "1 / minute"),  # scenario assumption
    acceptance_drop=0.10,  # scenario assumption
)


BurstyModelServerQueuePrice = _st.KingmanQueueScenario(
    name="One bursty model server at 70 and 90 percent utilization",
    description="1 s mean service, c_a^2 = 2 and c_s^2 = 4, held resources at $0.002 per second.",
    service=Q_(1, "second"),  # scenario assumption
    arrival_scv=2.0,  # scenario assumption
    service_scv=4.0,  # scenario assumption
    low_utilization=0.70,  # scenario assumption
    high_utilization=0.90,  # scenario assumption
    holding_rate=Q_(0.002, "USD / second"),  # scenario assumption
)


ThirtyCallTailExposure = _st.SerialTailExposureScenario(
    name="Thirty serial model calls, each past its own 99th percentile with probability 0.01",
    description="Independent per-call tail events along one trajectory.",
    calls=30,  # scenario assumption: serial calls on the path
    tail_probability=0.01,  # the definition of a 99th percentile
)


DedicatedCapacityBreakEven = _st.DedicatedBreakEvenScenario(
    name="Renting a matched-capability API against owning capacity at $28 an hour",
    description="Five input tokens per output token, 80 percent of input cached at $3, $0.30, and $15 "
    "per million; owned capacity at $28 an hour plus $2 per million output, 1,200 output "
    "tokens per second at full load, run at 52 percent.",
    input_per_output=5.0,  # scenario assumption
    cached_fraction=0.80,  # scenario assumption
    input_price=OutputCeilingThreeAllocations.input_price,
    cached_price=OutputCeilingThreeAllocations.input_price * RepairPrefixLayoutBill.cached_fraction,
    output_price=ProbeVsGeneratedExplanation.output_price,
    fixed_per_hour=Q_(28.0, "USD / hour"),  # scenario assumption: dedicated commitment
    variable_price=Q_(2.0, "USD / Mtoken"),  # scenario assumption: marginal cost per output token
    reference_rate=Q_(1_200, "1 / second"),  # scenario assumption: output tokens per second at full load
    load=0.52,  # scenario assumption: operating load
)


RepairBudgetLateBill = _st.ConservedBudgetScenario(
    name="A $5 repair budget across two children, one stopped with a remote job billed late",
    description="Planning settles $0.042; a trace child gets $1.20 and closes at $0.38; a fuzzing "
    "child gets $2.50, settles $0.85, and stops with $1 reserved for a remote job that "
    "bills $0.70 late, delivered twice.",
    budget=Q_(5.0, "USD"),  # scenario assumption
    planning=Q_(0.042, "USD"),  # scenario assumption
    allocations=[Q_(1.20, "USD"), Q_(2.50, "USD")],  # scenario assumption
    settled_children=[Q_(0.38, "USD"), Q_(0.85, "USD")],  # scenario assumption
    pending=Q_(1.0, "USD"),  # scenario assumption: enforceable maximum remaining remote charge
    late_charge=Q_(0.70, "USD"),  # scenario assumption: authenticated final bill
    deliveries=2,  # scenario assumption: the final bill is redelivered once
)


DependencyUpgradeArchitectures = _st.ArchitectureChoiceScenario(
    name="Dependency upgrades: a rule-based rewriter, one frontier call, or a bounded three-turn loop",
    description="10,000 upgrades a month; a $0.002 check per candidate; rejects go to a $35 engineer "
    "who resolves 90 percent; a 95 percent combined resolution floor.",
    tickets=10_000,  # scenario assumption: monthly arrivals
    single_costs={
        "Rule-based rewriter": Q_(0.001, "USD"),  # scenario assumption
        "Single model call": Q_(0.045, "USD"),
    },  # scenario assumption
    single_acceptance={
        "Rule-based rewriter": 0.32,  # scenario assumption
        "Single model call": 0.68,
    },  # scenario assumption
    loop_name="Bounded 3-turn loop",
    turn_cost=Q_(0.006, "USD"),  # scenario assumption
    turn_acceptance=[0.55, 0.42, 0.30],  # scenario assumption: conditional on reaching the turn
    check_cost=Q_(0.002, "USD"),  # scenario assumption
    human_cost=Q_(35, "USD"),  # scenario assumption: one fallback with its final check
    human_resolution=0.90,  # scenario assumption
    resolution_floor=0.95,  # scenario assumption
)


CodeReviewAdaptationPayback = _st.AdaptationPaybackScenario(
    name="Adapting a smaller model for code review, repaid by cheaper tasks",
    description="Five steps of 12,000 input and 800 output tokens at $3 and $15 per million, against "
    "four steps of 1,500 and 800 at $1 and $4; $4,000 of data, $500 of training, $8,000 of "
    "evaluation; 20,000 reviews a month, or a quarter of that.",
    base={
        "input_tokens": 12_000,
        "output_tokens": 800,
        "steps": 5,  # scenario assumption
        "input_price": OutputCeilingThreeAllocations.input_price,
        "output_price": ProbeVsGeneratedExplanation.output_price,
    },
    adapted={
        "input_tokens": 1_500,
        "output_tokens": 800,
        "steps": 4,  # scenario assumption
        "input_price": Q_(1, "USD / Mtoken"),  # scenario assumption
        "output_price": Q_(4, "USD / Mtoken"),
    },  # scenario assumption
    data_cost=Q_(4_000, "USD"),  # scenario assumption: collection and curation
    training_cost=Q_(500, "USD"),  # scenario assumption: training compute
    evaluation_cost=Q_(8_000, "USD"),  # scenario assumption: evaluation and review
    monthly_volume=20_000,  # scenario assumption
    low_volume=5_000,  # scenario assumption: a quarter of the planned volume
)


LighthouseWorkloadLevers = _st.WorkloadLeverScenario(
    name="Cost per accepted task for repair, research, remediation, and support, and three levers",
    description="Repair's tokens are the cached 30-turn bill with $0.03 of sandbox and tools per turn; "
    "research is the orchestrator-worker fan-out of a chat; remediation and support read "
    "their workload profiles. Levers: halve the token price, halve the largest term, or "
    "add 10 points of acceptance.",
    workloads={
        "repair": {
            "bill": RepairResendCacheBill,
            "turns": _Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
            "environment_per_turn": 0.03,  # scenario assumption: sandbox holding and tools, USD
            "assessment": 0.10,  # scenario assumption: sealed regression suite, USD
            "human": 0.0,
            "acceptance": 0.40,
        },  # scenario assumption
        "research": {
            "chat_input": _Agents.Research.OrchestratorWorker.chat_input_tokens.to("token").magnitude,
            "chat_output": _Agents.Research.OrchestratorWorker.chat_output_tokens.to("token").magnitude,
            "token_multiple": _Agents.Research.OrchestratorWorker.fanout_token_multiple,
            "environment": _Agents.Research.OrchestratorWorker.environment_cost_per_attempt.to("USD").magnitude,
            "assessment": _Agents.Research.OrchestratorWorker.assessment_cost_per_attempt.to("USD").magnitude,
            "human": 0.0,
            "acceptance": _Agents.Research.OrchestratorWorker.acceptance_probability,
        },
        "remediation": {
            "model": _Agents.Remediation.OnCall.model_cost_per_attempt.to("USD").magnitude,
            "environment": _Agents.Remediation.OnCall.environment_cost_per_attempt.to("USD").magnitude,
            "assessment": _Agents.Remediation.OnCall.assessment_cost_per_attempt.to("USD").magnitude,
            "human": _Agents.Remediation.OnCall.approval_cost_per_attempt.to("USD").magnitude,
            "acceptance": _Agents.Remediation.OnCall.acceptance_probability,
        },
        "support": {
            "model": _Agents.Support.CustomerOps.model_cost_per_attempt.to("USD").magnitude,
            "environment": _Agents.Support.CustomerOps.environment_cost_per_attempt.to("USD").magnitude,
            "assessment": _Agents.Support.CustomerOps.assessment_cost_per_attempt.to("USD").magnitude,
            "handoff": _Agents.Support.CustomerOps.handoff_cost.to("USD").magnitude,
            "acceptance": _Agents.Support.CustomerOps.acceptance_probability,
        },
    },
    input_price=RepairResendCacheBill.input_price,
    output_price=RepairResendCacheBill.output_price,
    acceptance_gain=0.10,  # scenario assumption: ten points of acceptance
)
