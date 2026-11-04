"""Named reliability scenarios for durability.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ..agents.agents import HorizonCompounding30Steps, RepairAttemptCostLevers
from ..agents.model_calls import Llama3_70B_Int8DecodeH100


# --- Durable execution (added 2026-10-20) ---------------------------------
# One trajectory's crash-recovery costs: a group commit, full snapshots
# against deltas, checkpoint strides on a fixed crash trace, Young's interval
# for re-executed work, the cold prefill of a resumed context, a rerun's
# reproduction over the step budget, and a year of recovery data.
IntentQuorumGroupCommit = _st.GroupCommitScenario(
    name="Sixteen intents acknowledged by one quorum flush",
    description="A 2 ms batch timer gathers 16 intents; one 4 ms quorum acknowledgment covers "
    "them all. Scenario timings, not measurements of a particular store.",
    batch_timer=Q_(2.0, "ms"),  # scenario assumption: longest wait before the batch starts
    flush_time=Q_(4.0, "ms"),  # scenario assumption: shared quorum acknowledgment
    batch_size=16,  # scenario assumption: intents per acknowledgment
)


GrowingTrajectorySnapshots = _st.SnapshotVolumeScenario(
    name="Full snapshots against deltas over a 500-turn trajectory",
    description="State grows 128 KB per turn and a checkpoint is taken every 5 turns.",
    turns=500,  # scenario assumption: trajectory length
    stride=5,  # scenario assumption: checkpoint every 5 turns
    state_growth=Q_(128, "kB"),  # scenario assumption: state growth per turn, decimal kilobytes
)


TwoCrashCheckpointTrace = _st.CheckpointTraceScenario(
    name="Checkpoint strides of 10, 25, and 50 turns on a 100-turn trace with two crashes",
    description="Crashes after turns 37 and 83; a 0.5 s pause per checkpoint, 1 s restore per "
    "crash, 0.1 s to rebuild one logged turn, 20 s to re-execute an unlogged one.",
    turns=100,  # scenario assumption
    crashes=[37, 83],  # scenario assumption: after these completed turns
    strides=[10, 25, 50],  # scenario assumption: the three policies compared
    pause=Q_(0.5, "s"),  # scenario assumption: serial pause including durable publication
    restore=Q_(1.0, "s"),  # scenario assumption: per crash
    rebuild_per_turn=Q_(0.1, "s"),  # scenario assumption: read, validate, fold one logged turn
    reexecute_per_turn=Q_(20.0, "s"),  # scenario assumption: re-run one turn's model call and tools
)


ReexecutedWorkCheckpointInterval = _st.CheckpointIntervalScenario(
    name="Young interval for re-executed agent work: a 2 s pause, a failure every 10 minutes",
    description="Work lost to a crash must be re-executed; the pause includes durable publication.",
    write_time=Q_(2.0, "s"),  # scenario assumption: pause per checkpoint
    mtbf=Q_(10, "minute"),  # scenario assumption: mean time between work-losing failures
    miss_factors=[2, 10],  # scenario assumption: an interval off by 2 and by an order of magnitude
)


ResumedContextColdPrefill = _st.ModelCallScenario(
    name="Cold prefill of a 32,768-token resumed context",
    description="No prefix state survives the resume; the whole recovered context is "
    "prefilled at the 8-bit compute floor before the next reply.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt_tokens=32_768,  # scenario assumption: recovered context of a long trajectory
    output_tokens=0,  # the floor before the first output token
)


GreedyRerunReproduction = HorizonCompounding30Steps.variant(
    name="A greedy rerun that reproduces each recorded reply 99 percent of the time",
    description="Each call of a temperature-zero rerun independently matches its recording; the "
    "rerun reproduces only if all calls over the coding-agent step budget match.",
    step_success=0.99,  # scenario assumption: per-call reproduction rate
)


RepairTrajectoryRetention = _st.RetentionCostScenario(
    name="A year of recovery and replay data for one repair trajectory",
    description="Thirty turns of 4 KB event envelopes and 128 KB of delivered inputs and "
    "results, one 50 MB workspace snapshot, kept 12 months in standard object "
    "storage, against the cost of one repair attempt.",
    turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    envelope=Q_(4, "kB"),  # scenario assumption: event envelopes per turn
    payload=Q_(128, "kB"),  # scenario assumption: delivered inputs and result bodies per turn
    workspace=Q_(50, "MB"),  # scenario assumption: one immutable workspace snapshot
    rate=_Infrastructure.Pricing.Storage.S3StandardPerTbMonth.rate,
    period=Q_(12, "month"),  # scenario assumption: promised replay horizon
    attempt_cost=RepairAttemptCostLevers.model_cost + RepairAttemptCostLevers.other_cost,
)
