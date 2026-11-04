"""Named training scenarios for fine tuning.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.types import TransformerWorkload as _TransformerWorkload
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ....core.units import BYTES_ADAM_MOMENTS as _BYTES_ADAM_MOMENTS
from ..agents.model_calls import CodingAgentResend30Turns


# --- Fine-tuning on trajectories (added 2026-10-20) ------------------------
# The repair agent's recovery record turned into training examples, a weight
# update, and a release decision. Token counts, log-probabilities, prices,
# and trial outcomes are hypothetical stipulations, not measurements.
RepairTrajectoryPerCallExamples = _st.PerInvocationSerializationScenario(
    name="A repair trajectory stored as one training example per model call",
    description="The repair harness's step budget in turns, each turn adding about the same tokens; "
    "every call's delivered prefix stored separately against one transcript.",
    turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    tokens_per_turn=CodingAgentResend30Turns.appended,
)


RecoveryRecordTargetMask = _st.TargetMaskScenario(
    name="One serialized repair recovery record and its selected targets",
    description="16,384 stored tokens: 1,200 of prompt, 184 of task, 1,400 of reviewed rationale, 960 "
    "of policy response, and 12,640 of observation; the refused proposal's 160 response "
    "tokens stay context.",
    prompt=1_200,  # scenario assumption: includes the initial boundary position
    task=184,  # scenario assumption
    rationale=1_400,  # scenario assumption: available, authorized, reviewed text
    response=960,  # scenario assumption: response content and authentic stop tokens
    observation=12_640,  # scenario assumption
    unselected_response=160,  # scenario assumption: the refused proposal, kept as context
)


RecoveryRecordTargetMaskDoubledObservations = RecoveryRecordTargetMask.variant(
    name="The same recovery record with its observations doubled",
    description="Every other segment unchanged.",
    observation=2 * RecoveryRecordTargetMask.observation,  # scenario assumption: doubled observations
)


RecoveryRecordTargetMaskLongObservations = RecoveryRecordTargetMask.variant(
    name="The same recovery record with 30,000 observation tokens",
    description="Every other segment unchanged.",
    observation=30_000,  # scenario assumption: longer observations
)


RepairTraceWithLookupWeighting = _st.LossNormalizationScenario(
    name="A long repair trace and a one-turn file lookup in one batch",
    description="8,192 selected targets against 256: token averaging against example averaging.",
    focus_targets=8_192,  # scenario assumption
    other_examples=1,  # scenario assumption
    other_targets=256,  # scenario assumption
)


EightExamplePackingBatch = _st.SequencePackingScenario(
    name="Eight independent examples, 512 to 32,768 tokens, against 32,768-token buffers",
    description="Padding every row to the longest against first-fit decreasing packing.",
    lengths=[32_768, 8_192, 4_096, 2_048, 1_024, 1_024, 512, 512],  # scenario assumption
    capacity=32_768,  # scenario assumption: buffer positions
)


RepairPolicyImitationBounds = _st.ImitationBoundScenario(
    name="Imitation cost bounds over the repair harness's 30-turn horizon",
    description="A 2 percent disagreement rate on reference states against the same rate on learner "
    "states with deviation disadvantage u = 1.",
    horizon=RepairTrajectoryPerCallExamples.turns,
    per_step_error=0.02,  # scenario assumption
    recoverability=1.0,  # scenario assumption: one slip costs at most one unit
)


RepairPreferencePair = _st.PreferencePairScenario(
    name="Two passing repair continuations scored against a frozen reference",
    description="Natural-log probability sums over the selected tokens: preferred -8 against a "
    "reference -10, dispreferred -11 against -10, beta 0.2.",
    policy_preferred=-8.0,  # scenario assumption
    reference_preferred=-10.0,  # scenario assumption
    policy_dispreferred=-11.0,  # scenario assumption
    reference_dispreferred=-10.0,  # scenario assumption
    beta=0.2,  # scenario assumption
)


RepairPreferencePairDoubledBeta = RepairPreferencePair.variant(
    name="The same preference pair at twice the beta",
    description="Identical log-probabilities; only the preference scale changes.",
    beta=2 * RepairPreferencePair.beta,  # scenario assumption: doubled preference scale
)


Rank16AdaptersOnHypothetical8B = _st.TrainingStateFitScenario(
    name="Rank-16 low-rank adapters on a hypothetical 8B-parameter transformer, mixed-precision Adam",
    description="Two 4,096-by-4,096 matrices adapted in each of 32 layers; 2 B per frozen parameter, "
    "16 B per trained parameter, before activations.",
    workload=_TransformerWorkload(
        name="Hypothetical 8B-parameter transformer",
        architecture="Transformer",
        parameters=Q_(8e9, "param"),  # scenario assumption: exact illustrative size
        layers=32,  # scenario assumption
        hidden_dim=4_096,  # scenario assumption
    ),
    bytes_per_value=_BYTES_FP16,
    optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
    master_bytes_per_parameter=_BYTES_FP32,
    adapter_rank=16,  # scenario assumption
    adapted_matrices=2,  # scenario assumption: matrices adapted per layer
    adapter_in_dim=4_096,  # scenario assumption
    adapter_out_dim=4_096,  # scenario assumption
)


Rank32AdaptersOnHypothetical8B = Rank16AdaptersOnHypothetical8B.variant(
    name="The same adapters at rank 32",
    description="Adapter state is linear in the rank; the frozen base is unchanged.",
    adapter_rank=2 * Rank16AdaptersOnHypothetical8B.adapter_rank,  # scenario assumption: doubled rank
)
