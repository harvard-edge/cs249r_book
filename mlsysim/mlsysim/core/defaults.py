"""Simulation parameters and tuneable defaults.

These are reasonable starting points for analytical modeling — override them
for your specific scenario. Every value cites its source.
"""

from .units import USD
from .provenance import TraceableConstant

# --- Reliability (Component MTTF) ---
# Mean Time To Failure for datacenter-grade components.
# Source: Meta (2024), Google (2024), Barroso et al. (2018)
GPU_MTTF_HOURS = TraceableConstant(
    50_000,
    name="GPU Mean Time To Failure",
    description="Steady-state MTTF for datacenter-grade GPU in continuous operation.",
    citation="Meta (2024), Google (2024), Barroso et al. (2018)"
)
NIC_MTTF_HOURS = 150_000           # Network interface card
PSU_MTTF_HOURS = 100_000           # Power supply unit
PCIE_SWITCH_MTTF_HOURS = 200_000   # PCIe switch/bridge
CABLE_MTTF_HOURS = 50_000          # Optical cable / transceiver (lowered for SDC analysis)
TOR_SWITCH_MTTF_HOURS = 300_000    # Top-of-rack switch
HBM_MTTF_HOURS = 200_000           # HBM memory module

# Silent Data Corruption (SDC) Assumptions
P_SDC_PER_GPU_HR = 1e-6

# Recovery time assumptions (seconds)
HEARTBEAT_TIMEOUT_S = 30            # Failure detection latency
RESCHEDULE_TIME_S = 60              # Time to allocate replacement node
CHECKPOINT_WRITE_BW_GBS = 100       # Aggregate storage write BW for checkpoints (GB/s)

# --- Cluster Scale References ---
# Canonical cluster sizes used as worked examples throughout Volume II.
CLUSTER_SMALL_GPUS = 256
CLUSTER_MEDIUM_GPUS = 2_048
CLUSTER_LARGE_GPUS = 8_192
CLUSTER_MEGA_GPUS = 100_000

# --- Inter-Node Network (Fleet-Scale Byte Rates) ---
# Byte-per-second equivalents for bandwidth calculations.
# These complement the Gbps values defined in units.py for bit-rate contexts.
INFINIBAND_NDR_BW_GBS = 50         # 400 Gbps / 8 = 50 GB/s per port
INFINIBAND_HDR_BW_GBS = 25         # 200 Gbps / 8 = 25 GB/s per port
INFINIBAND_XDR_BW_GBS = 100        # 800 Gbps / 8 = 100 GB/s per port (2025)
ETHERNET_400G_BW_GBS = 50          # 400 GbE = 50 GB/s
ETHERNET_800G_BW_GBS = 100         # 800 GbE = 100 GB/s (2025)
ROCE_100G_BW_GBS = 12.5            # 100 GbE RoCE = 12.5 GB/s

# Communication model parameters (α-β model)
IB_NDR_LATENCY_US = 5              # InfiniBand NDR one-way latency (μs)
IB_HDR_LATENCY_US = 7              # InfiniBand HDR one-way latency (μs)
ROCE_LATENCY_US = 10               # RoCE v2 one-way latency (μs)
TCP_LATENCY_US = 50                # TCP/IP over Ethernet one-way latency (μs)

# --- Sustainability ---
# Power Usage Effectiveness (PUE) — total facility power / IT equipment power
PUE_LIQUID_COOLED = TraceableConstant(
    1.06,
    name="PUE (Liquid-Cooled)",
    description="Best-in-class liquid-cooled AI datacenter PUE.",
    citation="Google Sustainability Report (2023)"
)
PUE_BEST_AIR = TraceableConstant(
    1.12,
    name="PUE (Best Air-Cooled)",
    description="Best-in-class air-cooled hyperscale datacenter PUE.",
    citation="Google Sustainability Report (2023)"
)
PUE_TYPICAL = TraceableConstant(
    1.40,
    name="PUE (Industry Average)",
    description="Industry average traditional datacenter PUE.",
    citation="Uptime Institute (2023), Global Data Center Survey"
)
PUE_LEGACY = 1.58                  # Older enterprise datacenters

# Water Usage Effectiveness (WUE) — liters per kWh
WUE_AIR_COOLED = 0.5               # Air-cooled (minimal water)
WUE_EVAPORATIVE = 1.8              # Evaporative cooling towers
WUE_LIQUID = 0.0                   # Closed-loop liquid cooling (near zero)

# Regional carbon intensity (gCO2 per kWh) — Source: IEA (2023)
CARBON_US_AVG_GCO2_KWH = TraceableConstant(
    429,
    name="Carbon Intensity (US Average)",
    description="US national average grid carbon intensity in gCO2/kWh.",
    citation="IEA (2023), World Energy Outlook"
)
CARBON_EU_AVG_GCO2_KWH = 270       # EU average grid
CARBON_QUEBEC_GCO2_KWH = TraceableConstant(
    20,
    name="Carbon Intensity (Quebec)",
    description="Quebec grid carbon intensity in gCO2/kWh (hydroelectric dominant).",
    citation="IEA (2023), World Energy Outlook"
)
CARBON_FRANCE_GCO2_KWH = 50        # France (nuclear dominant)
CARBON_IOWA_GCO2_KWH = 680         # Iowa reference mix used in MLSys·im carbon tutorials
CARBON_POLAND_GCO2_KWH = 820       # Poland (coal dominant)
CARBON_NORWAY_GCO2_KWH = 10        # Norway (hydroelectric)

# Power density
RACK_POWER_TRADITIONAL_KW = 12     # Traditional datacenter rack (kW)
RACK_POWER_AI_TYPICAL_KW = 70      # AI cluster rack, current generation (kW)
RACK_POWER_AI_HIGH_KW = 100        # AI cluster rack, high-density (kW)
AIR_COOLING_LIMIT_KW = 30          # Approximate rack power where air cooling fails (kW)

# --- MFU and Scaling Efficiency References ---
# Model FLOPS Utilization (MFU) — actual FLOPS / peak FLOPS
MFU_TRAINING_LOW = TraceableConstant(
    0.30,
    name="MFU Training (Lower Bound)",
    description="Lower bound MFU for well-optimized large-model training.",
    citation="Chowdhery et al. (2022), PaLM; Narayanan et al. (2021), Megatron-LM"
)
MFU_TRAINING_HIGH = TraceableConstant(
    0.50,
    name="MFU Training (Upper Bound)",
    description="Upper bound MFU for excellent large-model training runs.",
    citation="Chowdhery et al. (2022), PaLM"
)
MFU_INFERENCE_BATCH1 = TraceableConstant(
    0.05,
    name="MFU Inference (Batch 1)",
    description="MFU for single-request inference, heavily memory-bandwidth-bound.",
    citation="Pope et al. (2023), LLM Inference"
)
MFU_INFERENCE_BATCHED = 0.40       # Inference at large batch size

# --- Software Tax ---
# Latency overhead for a single kernel launch on a modern GPU.
# Source: NVIDIA (2024), "CUDA C++ Programming Guide."
KERNEL_LAUNCH_LATENCY_US = 15.0    # 15 μs typical launch overhead
FRAMEWORK_LAYER_TAX_MS = 0.01      # 10 μs typical framework tax per model layer (assumes graph compilation/fused kernels)

# Scaling efficiency η = T_1 / (N × T_N)
SCALING_EFF_32GPU = 0.90           # Near-linear regime
SCALING_EFF_256GPU = 0.70          # Communication starts to bite
SCALING_EFF_1024GPU = 0.50         # Significant overhead
SCALING_EFF_8192GPU = TraceableConstant(
    0.35,
    name="Scaling Efficiency (8192 GPUs)",
    description="Empirical scaling efficiency at fleet-scale (8192 GPUs).",
    citation="Empirical reference; varies by workload and network"
)

# Overhead budgets (fraction of wall time)
OVERHEAD_PIPELINE_BUBBLE = 0.05    # ~5% for well-tuned pipeline parallelism
OVERHEAD_CHECKPOINT = 0.03         # ~3% for optimized async checkpointing
OVERHEAD_FAILURE_RECOVERY = 0.10   # ~10% for failure and restart at 10K+ scale
OVERHEAD_MAINTENANCE = 0.05        # ~5% for rolling upgrades, maintenance windows

# --- Scaling Laws (Chinchilla Physics) ---
# Source: Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"
CHINCHILLA_TOKENS_PER_PARAM = TraceableConstant(
    20,
    name="Compute-Optimal Token Ratio",
    description="The optimal number of training tokens per model parameter (D ≈ 20P) to minimize loss for a given compute budget.",
    citation="Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.",
    url="https://arxiv.org/abs/2203.15556"
)

CHINCHILLA_COMPUTE_CONSTANT = TraceableConstant(
    6,
    name="Training Compute Constant (C ≈ 6PD)",
    description="The multiplier for calculating total training FLOPs. 2 FLOPs per parameter for the forward pass, and 4 FLOPs for the backward pass.",
    citation="Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.",
    url="https://arxiv.org/abs/2203.15556"
)

# --- Critical Batch Size (McCandlish et al. 2018) ---
# Source: McCandlish et al. (2018), "An Empirical Model of Large-Batch Training"
# Estimates for when Data Parallelism hits diminishing returns.
CRITICAL_BATCH_SIZE_BERT = 256
CRITICAL_BATCH_SIZE_GPT3 = 4096
CRITICAL_BATCH_SIZE_DEFAULT = 1024

# --- Orchestration & Queueing (Little's Law) ---
# Typical cluster utilization targets and arrival rates for scenarios.
TARGET_CLUSTER_UTILIZATION = 0.80  # 80% is high for shared research clusters
QUEUE_DISCIPLINE = "FIFO"          # First-In-First-Out (Baseline)
AVERAGE_RESEARCHER_JOB_DAYS = 2.0  # Median job length in research clusters

# --- Economics defaults ---
# Source: Barroso et al. (2018)
ANNUAL_MAINTENANCE_RATIO = 0.05      # 5% of CapEx per year
GPU_UNIT_COST_H100 = 30000 * USD     # NVIDIA H100 SXM (2024 street price)
GPU_UNIT_COST_A100 = 15000 * USD     # NVIDIA A100 SXM (2024 street price)
GPU_UNIT_COST_B200 = 40000 * USD     # NVIDIA B200 (2025 estimated)

# Default electricity price — Source: AWS US baseline (2024)
DEFAULT_KWH_PRICE = 0.12             # USD per kWh

# --- Quantization accuracy deltas ---
# Source: Gholami et al. (2021) survey medians
QUANT_ACCURACY_DELTA_INT8 = -0.005   # Median INT8 PTQ accuracy drop
QUANT_ACCURACY_DELTA_INT4 = -0.025   # Median INT4 GPTQ accuracy drop
QUANT_ACCURACY_DELTA_FP8 = -0.002    # FP8 (negligible for most models)

# --- Pruning accuracy ---
# Source: Blalock et al. (2020) survey
PRUNING_ACCURACY_THRESHOLD = 0.5     # Sparsity threshold where degradation accelerates
PRUNING_MILD_DELTA = -0.001          # Accuracy delta below threshold
PRUNING_STEEP_COEFFICIENT = 0.01     # Coefficient for exponential degradation above threshold
PRUNING_STEEP_EXPONENT = 2.0         # Exponent for degradation curve

# --- EfficiencyModel MFU adjustment (heuristic calibrations) ---
# These are empirically calibrated caps, NOT derived from first principles.
# Source: Dao et al. (2022) reports ~2-4x speedup for FlashAttention.
# Source: Chowdhery et al. (2022) PaLM reports MFU 0.46-0.57 for large Transformers.
MFU_FLASH_ATTENTION = 0.75           # Calibrated MFU for FlashAttention attention layers
MFU_FLASH_ATTENTION_CAP = 0.85      # Practical maximum for FlashAttention
MFU_FFN_CAP = 0.60                   # Practical maximum for FFN (GEMM-dominated)
MFU_CONV_CAP = 0.55                  # Practical maximum for convolution (im2col+GEMM)
HFU_MFU_RATIO = 1.1                  # HFU ≈ 1.1 × MFU (Chowdhery et al. 2022, PaLM)

# --- InferenceScalingModel ---
# Heuristic: average tokens per CoT reasoning step.
# Varies widely (10-1000+); 50 is a moderate default.
TOKENS_PER_REASONING_STEP = 50

# --- ScalingModel ---
# Conservative sustained MFU for GPU-day-to-FLOP conversion.
REFERENCE_MFU_SUSTAINED = 0.40

# --- ResponsibleEngineeringModel (heuristic calibration) ---
# DP-SGD slowdown model: slowdown ≈ 1 + k/ε.
# NOT derived from Abadi et al. (2016) — calibrated to match reported
# slowdowns: ~3x at ε=1.0, ~1.2x at ε=10.0.
DP_SGD_SLOWDOWN_COEFFICIENT = 2.0

# --- Engine Default Overrides ---

# Default scaling efficiency for parallel clusters
DEFAULT_SCALING_EFFICIENCY = TraceableConstant(
    0.90,
    name="Scaling Efficiency (η)",
    description="The efficiency of parallel scaling. A value of 0.90 means 90% of theoretical linear speedup is achieved.",
    citation="Common industry rule-of-thumb for highly optimized clusters."
)

# Default communication overlap efficiency (e.g., Megatron-LM can overlap ~85% of communication)
DEFAULT_OVERLAP_EFFICIENCY = TraceableConstant(
    0.85,
    name="Communication Overlap Efficiency",
    description="The fraction of network communication time that can be successfully hidden behind compute operations.",
    citation="Shoeybi et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.",
    url="https://arxiv.org/abs/1909.08053"
)

# Default compute efficiency (MFU baseline)
DEFAULT_COMPUTE_EFFICIENCY = TraceableConstant(
    0.50,
    name="Baseline Model FLOPs Utilization (MFU)",
    description="A highly optimized large language model typically achieves around 50% MFU due to communication overhead and memory bandwidth constraints.",
    citation="Chowdhery et al. (2022). PaLM: Scaling Language Modeling with Pathways.",
    url="https://arxiv.org/abs/2204.02311"
)
