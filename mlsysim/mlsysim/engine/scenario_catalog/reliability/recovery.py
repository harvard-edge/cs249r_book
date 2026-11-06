"""Named reliability scenarios for recovery.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ..agents.deliberation import OutputCeilingThreeAllocations, ProbeVsGeneratedExplanation


# --- Failure recovery (added 2026-10-20) ---------------------------------
# The repair agent's temporary test environment and the remediation agent's
# fleet change. Durations, token counts, and prices are teaching
# assumptions at the hypothetical $3 and $15 per million tokens used across
# the agent scenarios; the remediation risk comes from its agent profile.
TestConfigRepairVsRebuild = _st.RepairBreakEvenScenario(
    name="One bounded repair of a rejected test configuration against compensating and rebuilding",
    description="Compensation 40.6 s and rebuild 46 s with 38,500 input and 3,100 output tokens, "
    "against one isolated repair call of 2,390 input and 180 output tokens: 0.09 s "
    "before output, 50 tokens/s, 0.4 s of validation; repair success assumed 0.5.",
    unwind_time=Q_(40.6, "second"),  # scenario assumption: compensation
    rebuild_time=Q_(46.0, "second"),  # scenario assumption: includes obtaining and validating a configuration
    rebuild_input=38_500,  # scenario assumption: rebuild model input tokens
    rebuild_output=3_100,  # scenario assumption: rebuild model output tokens
    repair_input=2_390,  # scenario assumption: entire isolated repair request
    repair_output=180,  # scenario assumption: repair output tokens
    startup_time=Q_(0.09, "second"),  # scenario assumption: pre-output startup, not time to first token
    output_rate=Q_(50, "token / second"),  # scenario assumption: decode rate
    validation_time=Q_(0.4, "second"),  # scenario assumption: local validation on either outcome
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=ProbeVsGeneratedExplanation.output_price,
    assumed_success=0.5,  # scenario assumption: policy comparison, not a measurement
)


RemediationCanaryPivotLoss = _st.IrreversibleLossScenario(
    name="The remediation agent's fleet rollout with canary checks before the pivot",
    description="Wrong-change probability, impact, canary catch rate, and revert fidelity from the "
    "on-call remediation profile; attempt cost is its model, environment, health-check, "
    "and approver costs; sensitivity to ten more points of catch rate.",
    workload=_Agents.Remediation.OnCall,
    catch_step=0.10,  # scenario assumption: catch-rate increment for the sensitivity
)


RepairLoopToTurnCeiling = _st.UndetectedLoopScenario(
    name="An append-only repair loop run to a 50-turn ceiling or stopped after turn 4",
    description="A 1,000-token first input; each turn appends 300 output tokens and an 800-token "
    "error observation; 10 s per turn; a 64,000-token combined capacity; no "
    "compaction or prefix caching.",
    first_input=1_000,  # scenario assumption: first looping input, tokens
    output_per_turn=300,  # scenario assumption
    observation_per_turn=800,  # scenario assumption: appended error observation
    turn_time=Q_(10, "second"),  # scenario assumption: complete serial turn
    ceiling=50,  # scenario assumption: turn ceiling counted from the loop's start
    stop_after=4,  # scenario assumption: intervention after this completed turn
    context_window=64_000,  # scenario assumption: combined input and output capacity
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=ProbeVsGeneratedExplanation.output_price,
)


ThreeLayerRetryAmplification = _st.RetryAmplificationScenario(
    name="Three retry layers, each with three attempts, under one failing call",
    description="The trajectory's recovery loop, an API gateway, and a client library, each "
    "allowing one try plus two retries on its own allowance.",
    layer_names=["agent loop", "gateway", "client library"],  # from the caller down
    attempts_per_layer=3,  # scenario assumption: one try plus two retries
)


BlockingCallBulkhead = _st.BulkheadAdmissionScenario(
    name="256 blocking calls to a stalled tool against 300 workers, capped at 64 and 32 queued",
    description="One burst, no completions during admission, no nested calls.",
    arrivals=256,  # scenario assumption: trajectories each submitting one blocking call
    workers=300,  # scenario assumption: shared pool
    worker_cap=64,  # scenario assumption: dependency cap on occupied workers
    queue_cap=32,  # scenario assumption: dependency queue
)


RetryCohortJitter = _st.RetryJitterScenario(
    name="128 retries released together: inside 2 ms, or jittered over 2 s",
    description="No later retries and no unrelated traffic.",
    retries=128,  # scenario assumption: eligible retries released at one time
    burst_window=Q_(2, "ms"),  # scenario assumption: stipulated synchronized window
    jitter_span=Q_(2, "second"),  # scenario assumption: uniform jitter span
)
