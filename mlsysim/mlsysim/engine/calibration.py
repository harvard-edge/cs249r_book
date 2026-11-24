"""Parameters for analytical solvers and the roofline engine.

These values tune ``mlsysim.engine.solvers`` models and ``mlsysim.engine.engine.Engine``
when callers omit explicit arguments. Use ``Literature.*``, ``Systems.*``, and
``Infrastructure.*`` for sourced domain reference values.
"""

from ..core.provenance import sourced
from ..core import provenance_catalog as pc
from ..physics.memory import optimizer_state_bytes

# Engine: software stack latency (CUDA programming guide order of magnitude).
# Per-layer host dispatch is zero by default because CUDA Graphs, XLA, and
# TorchInductor launch compiled step graphs rather than dispatching per layer;
# callers modeling eager Python execution pass layer_overhead_ms explicitly.
KERNEL_LAUNCH_LATENCY_US = 15.0
FRAMEWORK_LAYER_TAX_MS = 0.0

# InferenceScalingModel / ScalingModel reference defaults
TOKENS_PER_REASONING_STEP = 50
REFERENCE_MFU_SUSTAINED = 0.40

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
DEFAULT_SCALING_EFFICIENCY = sourced(0.90, pc.SCALING_EFFICIENCY_RULE_OF_THUMB, name="Scaling Efficiency (η)", description="Default parallel scaling efficiency (η = 90%).")
DEFAULT_OVERLAP_EFFICIENCY = sourced(0.85, pc.MEGATRON_OVERLAP, name="Communication Overlap Efficiency", description="Fraction of comm time hidden behind compute.")
DEFAULT_COMPUTE_EFFICIENCY = sourced(0.50, pc.PALM_MFU, name="Baseline MFU", description="Baseline model FLOPs utilization for large LLM training.")

# CheckpointModel: optimizer state accounting for checkpoints
CHECKPOINT_BYTES_PER_PARAM_ADAM = 14
CHECKPOINT_BYTES_PER_PARAM_SGD = 4

# TrainingMemoryModel: optimizer state (weights accounted separately in memory model)
TRAINING_OPTIMIZER_BYTES_ADAM = optimizer_state_bytes("adam").m_as("byte")
TRAINING_OPTIMIZER_BYTES_SGD = optimizer_state_bytes("sgd").m_as("byte")

# Fallback bandwidths for undefined hardware attributes
FALLBACK_STORAGE_BANDWIDTH_GB_S = 1.0
FALLBACK_INTERCONNECT_BANDWIDTH_GB_S = 100.0
# Engine offload path: host link when hardware.interconnect is unset
# (PCIe Gen5 x16 ~= 64 GB/s per direction). Distinct from the generic
# interconnect fallback above, which models node-to-node links.
FALLBACK_PCIE_BANDWIDTH_GB_S = 64.0
