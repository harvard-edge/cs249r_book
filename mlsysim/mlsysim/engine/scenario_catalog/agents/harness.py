"""Named agents scenarios for harness.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ..agents.agents import HorizonCompounding30Steps
from ..agents.context import RepairPrefixLayoutBill
from ..agents.deliberation import OutputCeilingThreeAllocations, ProbeVsGeneratedExplanation
from ..agents.model_calls import CodingAgentResend30Turns


# --- The agent harness (added 2026-10-20) ---
# The runtime around one coding trajectory: a rule kept per turn, a pause
# behind one call's output allowance (CanceledStreamDecodeSavings
# .allowance_decode_time), approval escrow, fair scheduling of model calls,
# admission of the next call, the budget meter, and ceilings read from
# public runs. Prices are hypothetical, not a provider schedule.
FreezeRuleOverStepBudget = HorizonCompounding30Steps.variant(
    name="A rule the model keeps on 99 percent of turns, over the coding-agent step budget",
    description="A stated freeze honored independently on each turn; the target is that 99 "
    "percent of runs keep it on every turn.",
    step_success=0.99,  # scenario assumption: hypothetical per-turn compliance
    target_success=0.99,  # scenario assumption: share of runs that must keep the rule throughout
)


RepairRemediationApprovalEscrow = _st.ApprovalEscrowScenario(
    name="Approval escrow for a repair agent and an on-call remediation agent",
    description="Repair review requests at 6 per hour with 2 h mean decisions, each holding the "
    "SWE-bench coding harness's workspace lease; remediation at the on-call "
    "profile's request rate, diagnosis, decision, and apply times.",
    repair_rate=Q_(6, "1/hour"),  # scenario assumption: review requests per hour
    repair_wait=Q_(2, "hour"),  # scenario assumption: mean time to a review decision
    lease=_Agents.Coding.SWE_Bench_Runner.sandbox_memory_footprint,
    remediation_rate=_Agents.Remediation.OnCall.approval_request_rate,
    diagnosis=_Agents.Remediation.OnCall.diagnosis_time,
    remediation_wait=_Agents.Remediation.OnCall.approval_decision_time,
    apply_verify=_Agents.Remediation.OnCall.apply_verify_time,
)


InteractiveMaintenanceDrr = _st.DeficitRoundRobinScenario(
    name="Interactive and maintenance agent calls under weighted deficit round-robin",
    description="Weights 3 and 1 and an 8,000-token quantum per unit weight per round; an "
    "interactive call charges 2,000 input tokens plus a 500-token output "
    "allowance, a maintenance call 40,000 plus 2,000.",
    quantum=8000,  # scenario assumption: charge tokens per unit weight per round
    classes={
        "interactive": {"weight": 3, "input": 2000, "output": 500},  # scenario assumption
        "maintenance": {"weight": 1, "input": 40000, "output": 2000},
    },  # scenario assumption
    simulated_rounds=10_000,  # rounds of the direct simulation that checks the formula
)


InteractiveMaintenanceDrrCompacted = InteractiveMaintenanceDrr.variant(
    name="The same schedule with maintenance prompts compacted to half their input",
    classes={
        **InteractiveMaintenanceDrr.classes,
        "maintenance": {
            **InteractiveMaintenanceDrr.classes["maintenance"],
            "input": InteractiveMaintenanceDrr.classes["maintenance"]["input"] // 2,
        },
    },
)


RepairNextCallAdmission = _st.BoundedLiabilityAdmissionScenario(
    name="The next model call of a repair trajectory against a $3.50 allowance",
    description="$2.58 settled, a $0.50 reservation on an unsettled test job, and a $0.30 "
    "finalization reserve; the call sends 50,000 input tokens with a 4,096-token "
    "output allowance at a hypothetical $3 and $15 per million tokens.",
    allowance=Q_(3.50, "USD"),  # scenario assumption: dollar allowance of the trajectory
    settled=Q_(2.58, "USD"),  # scenario assumption: actual spend so far
    reserved=Q_(0.50, "USD"),  # scenario assumption: reservation on the unsettled test job
    finalization_reserve=Q_(0.30, "USD"),  # scenario assumption: protected for checks and report
    input_tokens=50_000,  # scenario assumption: next call's input
    output_allowance=4096,  # scenario assumption: next call's output allowance
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=ProbeVsGeneratedExplanation.output_price,
    probe=Q_(0.10, "USD"),  # an unsettled amount used to express sensitivity
)


RepairTrajectoryBudgetMeter = _st.TrajectoryBudgetMeterScenario(
    name="The 30-turn coding trajectory metered against turns, tokens, time, and dollars",
    description="Allowances of 40 turns, 2 million tokens, 900 s, and the repair trajectory's "
    "$3.50, read at 460 s on the clock; hypothetical $3 input, $0.30 cached, and "
    "$15 output per million tokens.",
    trajectory=CodingAgentResend30Turns,
    input_price=OutputCeilingThreeAllocations.input_price,
    cached_fraction=RepairPrefixLayoutBill.cached_fraction,
    output_price=ProbeVsGeneratedExplanation.output_price,
    turn_allowance=40,  # scenario assumption
    token_allowance=2_000_000,  # scenario assumption
    time_allowance=Q_(900, "second"),  # scenario assumption
    dollar_allowance=RepairNextCallAdmission.allowance,
    elapsed=Q_(460, "second"),  # scenario assumption: wall clock at the readout
)
