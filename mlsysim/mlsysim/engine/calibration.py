"""Parameters for analytical solvers and the roofline engine.

These values tune ``mlsysim.engine.solver`` models and ``mlsysim.engine.engine.Engine``
when callers omit explicit arguments. Not appendix-facing — use ``Literature.*``,
``Systems.*``, and ``Infrastructure.*`` for book-cited numbers.
"""

from ..core.provenance import sourced
from ..core import provenance_catalog as pc

# Engine: software stack latency (CUDA programming guide order of magnitude)
KERNEL_LAUNCH_LATENCY_US = 15.0
FRAMEWORK_LAYER_TAX_MS = 0.01

# EfficiencyModel / InferenceScalingModel / ResponsibleEngineeringModel
MFU_FLASH_ATTENTION = 0.75
MFU_FLASH_ATTENTION_CAP = 0.85
MFU_FFN_CAP = 0.60
MFU_CONV_CAP = 0.55
HFU_MFU_RATIO = 1.1
TOKENS_PER_REASONING_STEP = 50
REFERENCE_MFU_SUSTAINED = 0.40
DP_SGD_SLOWDOWN_COEFFICIENT = 2.0

# ScalingModel: Reference FLOPS for GPU-days conversions
REFERENCE_HARDWARE_TFLOPS = 989.0  # NVIDIA H100 SXM FP16 Dense

# SustainabilityModel: energy proportionality (standard datacenter rules of thumb)
ENERGY_IDLE_FRACTION = 0.30
ENERGY_DYNAMIC_FRACTION = 0.70

# ServingModel: speculative decoding defaults
SPECULATIVE_GAMMA = 4  # Typical lookahead (K) for draft verification

# EconomicsModel: default multipliers for TCO
DEFAULT_INFRASTRUCTURE_MULTIPLIER = 1.0  # Hardware only. Use 2.0-2.5 for full DC.

# DistributedModel API fallbacks when efficiency kwargs are omitted
DEFAULT_SCALING_EFFICIENCY = sourced(0.90, pc.BOOK_SCALING_RULE_OF_THUMB, name="Scaling Efficiency (η)", description="Default parallel scaling efficiency (90%).")
DEFAULT_OVERLAP_EFFICIENCY = sourced(0.85, pc.MEGATRON_OVERLAP, name="Communication Overlap Efficiency", description="Fraction of comm time hidden behind compute.")
DEFAULT_COMPUTE_EFFICIENCY = sourced(0.50, pc.PALM_MFU, name="Baseline MFU", description="Baseline model FLOPs utilization for large LLM training.")

# CompressionModel: quantization / pruning survey medians (Gholami 2021, Blalock 2020)
QUANT_ACCURACY_DELTA_INT8 = -0.005
QUANT_ACCURACY_DELTA_INT4 = -0.025
QUANT_ACCURACY_DELTA_FP8 = -0.002
QUANT_ACCURACY_DELTA_SUB_INT4 = -0.050
PRUNING_ACCURACY_THRESHOLD = 0.5
PRUNING_MILD_DELTA = -0.001
PRUNING_STEEP_COEFFICIENT = 0.01
PRUNING_STEEP_EXPONENT = 2.0

# CheckpointModel: optimizer state accounting for checkpoints
CHECKPOINT_BYTES_PER_PARAM_ADAM = 14
CHECKPOINT_BYTES_PER_PARAM_SGD = 4

# TrainingMemoryModel: optimizer state (weights accounted separately in memory model)
TRAINING_OPTIMIZER_BYTES_ADAM = 12.0  # FP32 master + first/second moments
TRAINING_OPTIMIZER_BYTES_SGD = 4.0    # FP32 master

# Fallback bandwidths for undefined hardware attributes
FALLBACK_STORAGE_BANDWIDTH_GB_S = 1.0
FALLBACK_INTERCONNECT_BANDWIDTH_GB_S = 100.0
