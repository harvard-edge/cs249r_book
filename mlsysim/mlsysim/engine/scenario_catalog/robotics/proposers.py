"""Named robotics scenarios for proposers.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st


# --- Learned proposers on the warehouse mobile manipulator (added 2026-10-20) ---
IntentModelEvaluationFp16 = _st.VlaEvaluationScenario(
    name="The warehouse manipulator's 7B intent model with 16-bit weights",
    description="One fresh evaluation: compute-bound prefill of one view plus prompt, then a "
    "bandwidth-bound weight sweep per remaining action token, on the Jetson AGX Orin.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    weight_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
)


IntentModelEvaluation8Bit = IntentModelEvaluationFp16.variant(
    name="The warehouse manipulator's 7B intent model with 8-bit weights",
    weight_bytes_per_param=Q_(1, "byte/param"),  # scenario assumption: 8-bit weights, 16-bit arithmetic
)


IntentModelEvaluationInt4 = _st.VlaEvaluationScenario(
    name="The warehouse manipulator's 7B intent model as deployed (INT4 weights)",
    description="The deployed configuration: weight-only INT4, one view plus prompt, one token per action dimension.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
)


IntentModelEvaluationParallel = IntentModelEvaluationInt4.variant(
    name="The deployed intent model with parallel action decoding",
    parallel_decode=True,  # scenario assumption: OpenVLA-OFT-style parallel decoding
)


ThreeViewVlaTokens = _st.VlaTokenBudgetScenario(
    name="Tokens of a three-camera VLA step on the intent model's backbone",
    description="Patch tokens of 224-pixel views at patch 14 (OpenVLA config.json), with prompt and "
    "joint-state tokens, and the embedding and key-value bytes they occupy.",
    model=_Embodied.MobileManipulator.WarehouseMobileManipulator.intent_model,
    views=3,  # scenario assumption: left wrist, right wrist, overhead
    frame_px=224,  # OpenVLA config.json image_sizes
    patch_px=14,  # OpenVLA config.json vision encoders (patch14)
    prompt_tokens=32,  # scenario assumption: instruction tokens
    state_tokens=16,  # scenario assumption: joint-state tokens
    context_tokens=800,  # scenario assumption: rounded context length S
    element_bytes=Q_(2, "byte"),  # scenario assumption: FP16 activations, keys, and values
)


IntentModelEvaluationThreeViews = IntentModelEvaluationInt4.variant(
    name="The deployed intent model reading three camera views",
    input_tokens=ThreeViewVlaTokens.context_tokens,
)


IntentModelFiveHertz = _st.RateTargetScenario(
    name="The deployed intent model against a 5 Hz target",
    description="One fresh evaluation per 200 ms period, not pipelined.",
    evaluation=IntentModelEvaluationInt4,
    target_rate=Q_(5, "Hz"),  # scenario assumption: candidate intent rate
)


IntentModelParallelFiveHertz = IntentModelFiveHertz.variant(
    name="Parallel action decoding against a 5 Hz target", evaluation=IntentModelEvaluationParallel
)


ThreeViewKvHistory = _st.KvHistoryScenario(
    name="Five seconds of retained three-camera context on the intent model's backbone",
    description="A hypothetical history-keeping variant: key-value cache at 10 frames a second beside "
    "the 16-bit weights, against the module's memory.",
    tokens=ThreeViewVlaTokens,
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    history_window=Q_(5, "second"),  # scenario assumption
    history_rate=Q_(10, "Hz"),  # scenario assumption
    weight_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
)


IntentBackboneDecoderFamilies = _st.DecoderFamiliesScenario(
    name="One fresh 16-step chunk from the INT4 intent backbone, per decoder family",
    description="Binned autoregressive, FAST, parallel, and flow-expert decoding after the shared prefill.",
    backbone=IntentModelEvaluationInt4,
    fast_tokens=30,  # Pertsch, K. et al. (2025), FAST: Efficient Action Tokenization for ...
    expert_params=Q_(300000000.0, "param"),  # Black, K. et al. (2024), pi0: A Vision-Language-Action Flow Model f...
    expert_steps=10,  # Black, K. et al. (2024), pi0: A Vision-Language-Action Flow Model f...
    expert_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: 16-bit expert weights
    kv_bytes_per_value=Q_(2, "byte"),  # scenario assumption: FP16 keys and values
)


WarehouseProposerTraffic = _st.ProposerTrafficScenario(
    name="The warehouse manipulator's two proposers on its memory bus",
    description="Intent-model sweeps, chunk supply, renewal gap, and the proposers' share of the sustained bandwidth.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    fp16_bytes_per_param=Q_(2, "byte/param"),  # scenario assumption: FP16 weights
)


ConveyorMugEvidenceHorizon = _st.EvidenceHorizonScenario(
    name="The tracked conveyor mug against the chunk policy's renewal",
    description="Untracked deadline, tracked evidence horizon, age at replacement, and the drift of a "
    "chunk executed to its end.",
    site=_Embodied.Scenario.WarehouseAisle,
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
)


ConveyorMugEvidenceHorizonHalfRenewal = ConveyorMugEvidenceHorizon.variant(
    name="The tracked conveyor mug with the chunk policy renewing at half rate",
    chunk_rate=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_rate / 2,  # scenario assumption
)


OpenVlaOrinQuantization = _st.WeightOnlyQuantizationScenario(
    name="OpenVLA-7B weight formats on the Jetson AGX Orin, computed against measured",
    description="The intent model priced at FP16, 8-bit, and INT4 weights against the Jetson AI Lab "
    "measurements and action scores of the same formats.",
    formats={"FP16": IntentModelEvaluationFp16, "8-bit": IntentModelEvaluation8Bit, "INT4": IntentModelEvaluationInt4},
    measured={
        "FP16": Q_(840.0, "ms"),  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
        "8-bit": Q_(471.0, "ms"),  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
        "INT4": _Embodied.MobileManipulator.WarehouseMobileManipulator.intent_reference_latency,
    },
    scores={
        "FP16": 0.953,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
        "8-bit": 0.952,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
        "INT4": 0.901,  # Franklin, D., NVIDIA Jetson AI Lab, 'OpenVLA' tutorial (2024): Open...
    },
    deployed="INT4",
    reference="FP16",
    group_size=128,  # Lin, J. et al. (2024), AWQ: Activation-aware Weight Quantization fo...
    scale_bits=16,  # scenario assumption: one FP16 scale per group
    weight_bits=4,  # scenario assumption: INT4 weights
)


ProposerConfidenceRegimes = _st.ConfidenceRegimesScenario(
    name="Stipulated proposer confidence against task success under distribution shift",
    description="Illustrative values for four operating regimes, not test results.",
    regimes=[  # scenario assumption: stipulated, illustrative (confidence, task success)
        ("In distribution", 0.91, 0.89),
        ("Near OOD", 0.89, 0.52),
        ("Sensor degraded", 0.84, 0.31),
        ("Far OOD", 0.78, 0.08),
    ],
)
