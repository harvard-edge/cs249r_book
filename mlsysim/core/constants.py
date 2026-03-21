# constants.py
# The Analytical Engine of Machine Learning Systems
# This file defines the single source of truth for hardware specifications,
# constants, and conversion factors used throughout the textbook.
#
# Measurement units live in units.py; tuneable simulation defaults live in
# defaults.py. This module re-exports both for backward compatibility and
# adds the genuine hardware/model constants that belong here.

from .units import *  # noqa: F401,F403 — re-export full unit registry

# --- Hardware Specifications (The Silicon Contract) ---

# NVIDIA V100 (Volta, 2017) — Source: NVIDIA V100 Data Sheet
V100_FLOPS_FP16_TENSOR = 125 * TFLOPs / second
V100_FLOPS_FP32 = 15.7 * TFLOPs / second
V100_MEM_BW = 900 * GB / second           # HBM2
V100_MEM_CAPACITY = 32 * GiB
V100_TDP = 300 * watt                     # SXM2 variant

# NVIDIA A100 (Ampere, 2020) — Source: NVIDIA A100 Data Sheet
A100_FLOPS_FP16_TENSOR = 312 * TFLOPs / second
A100_FLOPS_TF32 = 156 * TFLOPs / second
A100_FLOPS_FP32 = 19.5 * TFLOPs / second  # Standard CUDA cores
A100_FLOPS_INT8 = 624 * TFLOPs / second   # INT8 Tensor Core
A100_MEM_BW = 2039 * GB / second           # HBM2e (SXM variant)
A100_MEM_CAPACITY = 80 * GiB              # SXM variant (also 40 GiB PCIe)
A100_TDP = 400 * watt                     # SXM variant

# NVIDIA H100 (Hopper, 2022) — Source: NVIDIA H100 Data Sheet
H100_FLOPS_FP16_TENSOR = 989 * TFLOPs / second
H100_FLOPS_FP8_TENSOR = 1979 * TFLOPs / second
H100_FLOPS_TF32 = 494 * TFLOPs / second
H100_FLOPS_INT8 = 1979 * TFLOPs / second  # Dense. Sparse is 3958.
H100_MEM_BW = 3.35 * TB / second          # HBM3
H100_MEM_CAPACITY = 80 * GiB
H100_TDP = 700 * watt                     # SXM variant

# NVIDIA H200 (Hopper, 2023) — Source: NVIDIA H200 Data Sheet
# H200 shares the Hopper compute die with H100, only memory differs
H200_MEM_BW = 4.8 * TB / second             # HBM3e
H200_MEM_CAPACITY = 141 * GB
H200_TDP = 700 * watt                       # Same as H100 SXM

# NVIDIA B100/B200 (Blackwell, 2024) — Source: NVIDIA Blackwell Architecture
B200_FLOPS_FP16_TENSOR = 2250 * TFLOPs / second  # Dense. Sparse is 4500.
B200_FLOPS_FP16_SPARSE = 4500 * TFLOPs / second
B200_FLOPS_FP8_TENSOR = 4500 * TFLOPs / second   # Dense. Sparse is 9000.
B200_FLOPS_INT4 = 9000 * TFLOPs / second         # Dense. Sparse is 18 PFLOPS.
B200_MEM_BW = 8 * TB / second             # HBM3e
B200_MEM_CAPACITY = 192 * GiB
B200_TDP = 1000 * watt

# NVIDIA GB200 NVL72 (Rack-scale, 2024) — Source: NVIDIA Blackwell Architecture
# This is a full rack containing 72 Blackwell GPUs and 36 Grace CPUs.
# We model the aggregate resources of the rack for macro-scale simulation.
NVL72_GPUs = 72 * count
NVL72_FLOPS_FP8_TENSOR = 720 * PFLOPs / second  # 72 * 10 PFLOPS (Dense/Sparse vary)
NVL72_MEM_CAPACITY = 13.8 * TB                  # 72 * 192 GB
NVL72_MEM_BW = 576 * TB / second                # 72 * 8 TB/s
NVL72_NVLINK_BW = 130 * TB / second             # Full bisection (bidirectional)
NVL72_TDP = 120 * kilowatt                      # Max rack power
NVL72_UNIT_COST = 3000000 * USD                 # Estimated $3M+ per rack

# AMD Instinct MI300X (CDNA 3, 2023) — Source: AMD Instinct MI300X Data Sheet
MI300X_FLOPS_FP16_TENSOR = 1307 * TFLOPs / second  # Dense. Sparse is 2614.
MI300X_MEM_BW = 5.3 * TB / second
MI300X_MEM_CAPACITY = 192 * GiB
MI300X_TDP = 750 * watt

# NVIDIA T4 (Turing, 2018) — Source: NVIDIA T4 Data Sheet
T4_FLOPS_FP16_TENSOR = 65 * TFLOPs / second
T4_FLOPS_INT8 = 130 * TFLOPs / second
T4_MEM_BW = 320 * GB / second
T4_TDP = 70 * watt

# Google TPU v1 — Source: Jouppi et al. (2017)
TPUV1_FLOPS_INT8 = 92 * TFLOPs / second
TPUV1_TDP = 75 * watt

# Google TPU v2 — Source: Google Cloud Documentation
TPUV2_FLOPS_BF16 = 45 * TFLOPs / second
TPUV2_MEM_BW = 700 * GB / second
TPUV2_MEM_CAPACITY = 16 * GiB

# Google TPU v3 — Source: Google Cloud Documentation
TPUV3_FLOPS_BF16 = 105 * TFLOPs / second
TPUV3_MEM_BW = 900 * GB / second
TPUV3_MEM_CAPACITY = 32 * GiB

# Google TPU v4 — Source: Google TPUv4 paper (Jouppi et al., 2023)
TPUV4_FLOPS_BF16 = 275 * TFLOPs / second
TPUV4_MEM_BW = 1200 * GB / second

# Google TPU v5p — Source: Google Cloud Documentation (2024)
TPUV5P_FLOPS_BF16 = 459 * TFLOPs / second
TPUV5P_MEM_BW = 2.76 * TB / second
TPUV5P_MEM_CAPACITY = 95 * GiB
TPUV5P_ICI_BW = 1600 * GB / second        # Inter-Chip Interconnect

# Google TPU v6 (Trillium, 2024/25) — Source: Google Blog (Projected/Early)
TPUV6_FLOPS_BF16 = 2150 * TFLOPs / second  # ~4.7x over v5p (estimated peak)
TPUV6_MEM_BW = 4.5 * TB / second
TPUV6_MEM_CAPACITY = 128 * GiB

# Cerebras Wafer-Scale Engine (WSE) — Source: Cerebras Whitepapers
WSE1_CORES = 400000 * count
WSE1_MEM_CAPACITY = 18 * GB
WSE1_MEM_BW = 9 * PB / second
WSE1_TDP = 15000 * watt
WSE1_FLOPS_FP16 = 9 * PFLOPs / second  # Estimated

WSE2_CORES = 850000 * count
WSE2_MEM_CAPACITY = 40 * GB
WSE2_MEM_BW = 20 * PB / second
WSE2_TDP = 15000 * watt
WSE2_FLOPS_FP16 = 38 * PFLOPs / second  # Estimated

WSE3_CORES = 900000 * count
WSE3_MEM_CAPACITY = 44 * GB
WSE3_MEM_BW = 21 * PB / second
WSE3_TDP = 23000 * watt
WSE3_FLOPS_FP16 = 125 * PFLOPs / second

# High-end Desktop CPU (Reference)
CPU_FLOPS_FP32 = 1 * TFLOPs / second

# --- Latency Hierarchy (2025 Reference) ---
LATENCY_L1_REGISTER = 1 * NS
LATENCY_L2_CACHE = 4 * NS
LATENCY_HBM3 = 300 * NS
LATENCY_NVLINK = 500 * NS
LATENCY_PCIE_GEN5 = 1000 * NS
LATENCY_INFINIBAND = 5000 * NS
LATENCY_NVME_SSD = 100000 * NS

# Mobile NPU
MOBILE_NPU_TOPS_INT8 = 50 * TFLOPs / second
MOBILE_FLAGSHIP_NPU_TOPS_INT8 = 100 * TFLOPs / second
MOBILE_NPU_MEM_BW = 100 * GB / second

# --- Datasets ---
IMAGENET_IMAGES = 1_281_167 * count
IMAGENET_TEST_IMAGES = 50_000 * count
CIFAR10_IMAGES = 50_000 * count
CIFAR10_TEST_IMAGES = 10_000 * count

# Standard dimensions
IMAGE_DIM_RESNET = 224
IMAGE_CHANNELS_RGB = 3
COLOR_DEPTH_8BIT = 256

# --- Network & Interconnect ---
NETWORK_10G_BW = 10 * Gbps
NETWORK_100G_BW = 100 * Gbps
NETWORK_5G_ENERGY_PER_MB_MJ = 100 * ureg.millijoule / MB

# Optical Interconnects (2025-2026 Reference)
OPTICS_POWER_PLUGGABLE_400G_W = 20 * watt
OPTICS_POWER_CPO_400G_W = 10 * watt
OPTICS_POWER_LPO_400G_W = 12 * watt       # Linear Pluggable Optics

# Intra-node interconnects
NVLINK_V100_BW = 300 * GB / second        # NVLink 2.0 (V100, 6 links × 50 GB/s)
NVLINK_A100_BW = 600 * GB / second        # NVLink 3.0 (A100, 12 links × 50 GB/s)
NVLINK_H100_BW = 900 * GB / second        # NVLink 4.0 (H100, 18 links × 50 GB/s)
NVLINK_B200_BW = 1800 * GB / second       # NVLink 5.0 (B200, 72 links × 25 GB/s)
PCIE_GEN3_BW = 15.75 * GB / second        # PCIe Gen3 x16 (after 128b/130b encoding)
PCIE_GEN4_BW = 32 * GB / second           # PCIe Gen4 x16 (bidirectional)
PCIE_GEN5_BW = 64 * GB / second           # PCIe Gen5 x16 (bidirectional)

# Inter-node interconnects
INFINIBAND_HDR_BW = 200 * Gbps            # HDR InfiniBand (25 GB/s)
INFINIBAND_NDR_BW = 400 * Gbps            # NDR InfiniBand (50 GB/s)
INFINIBAND_XDR_BW = 800 * Gbps            # XDR InfiniBand (100 GB/s)
INFINIBAND_GXDR_BW = 1600 * Gbps          # GXDR InfiniBand (200 GB/s, 2026)

# --- Energy (Horowitz, 2014 @ 45nm) ---
ENERGY_DRAM_ACCESS_PJ = 640 * ureg.picojoule
ENERGY_DRAM_PJ_PER_BYTE = 160 * ureg.picojoule / byte
ENERGY_FLOP_FP32_PJ = 3.7 * ureg.picojoule / flop   # FP32 multiply-add
ENERGY_FLOP_FP16_PJ = 1.1 * ureg.picojoule / flop   # FP16 multiply-add
ENERGY_FLOP_INT8_PJ = 0.2 * ureg.picojoule / flop   # INT8 multiply-add
ENERGY_FLOP_PJ = 4.6 * ureg.picojoule / flop         # Generic (legacy alias)
ENERGY_SRAM_L1_PJ = 0.5 * ureg.picojoule             # L1 cache access
ENERGY_SRAM_L2_PJ = 2.0 * ureg.picojoule             # L2 cache access
ENERGY_REG_PJ = 0.01 * ureg.picojoule                # Register file access
ENERGY_MOBILENET_INF_MJ = 0.1 * ureg.millijoule

# Addition energy (Horowitz 2014, 45nm process)
ENERGY_ADD_FP32_PJ = 0.9 * ureg.picojoule
ENERGY_ADD_FP16_PJ = 0.4 * ureg.picojoule
ENERGY_ADD_INT32_PJ = 0.1 * ureg.picojoule
ENERGY_ADD_INT8_PJ = 0.03 * ureg.picojoule

# Network transfer energy (reference)
NETWORK_ENERGY_1KB_PJ = 1_000_000 * ureg.picojoule  # ~1 microjoule for 1KB

# --- Infrastructure & Grid ---
LEAD_TIME_GPU_MONTHS = 6
LEAD_TIME_SUBSTATION_MONTHS = 24
GRID_INTERCONNECTION_QUEUE_US_GW = 2000

# --- Physics ---
# --- Physical Constants ---
SPEED_OF_LIGHT_FIBER_KM_S = 200000 * ureg.kilometer / second

# --- Cloud Pricing ---
CLOUD_EGRESS_PER_GB = 0.09 * USD / GB  # AWS data transfer out (2024 baseline)
CLOUD_ELECTRICITY_PER_KWH = 0.12 * USD / ureg.kilowatt_hour

# Storage Pricing (2024 baseline)
STORAGE_COST_S3_STD = 23 * USD / TB / ureg.month
STORAGE_COST_GLACIER = 1 * USD / TB / ureg.month
STORAGE_COST_NVME_LOW = 100 * USD / TB / ureg.month
STORAGE_COST_NVME_HIGH = 300 * USD / TB / ureg.month
RETRIEVAL_COST_GLACIER = 0.02 * USD / GB

# Labeling Pricing (2024 estimates)
LABELING_COST_CROWD_LOW = 0.01 * USD
LABELING_COST_CROWD_HIGH = 0.05 * USD
LABELING_COST_EXPERT_LOW = 0.50 * USD
LABELING_COST_EXPERT_HIGH = 2.00 * USD
LABELING_COST_BOX_LOW = 0.05 * USD
LABELING_COST_BOX_HIGH = 0.20 * USD
LABELING_COST_SEG_LOW = 5 * USD
LABELING_COST_SEG_HIGH = 50 * USD
LABELING_COST_MEDICAL_LOW = 50 * USD
LABELING_COST_MEDICAL_HIGH = 200 * USD

# GPU pricing (scenario baselines)
CLOUD_GPU_TRAINING_PER_HOUR = 4.0 * USD / hour
CLOUD_GPU_INFERENCE_PER_HOUR = 2.5 * USD / hour
TPU_V4_PER_HOUR = 4.0 * USD / hour

# --- Carbon (Scenario Baseline) ---
CARBON_PER_GPU_HR_KG = 0.16 * ureg.kilogram

# --- Mobile / Battery ---
MOBILE_TDP_W = 3 * watt
PHONE_BATTERY_WH = 15 * watt * hour
OBJECT_DETECTOR_POWER_W = 2 * watt
SERVER_POWER_W = 300 * watt

# Reference energies
ENERGY_SMARTPHONE_CHARGE_J = 40000 * joule
ENERGY_BOILING_WATER_J = 100000 * joule

# --- Video ---
VIDEO_1080P_WIDTH = 1920
VIDEO_1080P_HEIGHT = 1080
VIDEO_BYTES_PER_PIXEL_RGB = 3 * byte
VIDEO_FPS_STANDARD = Q_(30, 'Hz')

# --- Models & Workloads ---

# GPT-2 (1.5B) — used in training chapter worked examples
GPT2_PARAMS = 1.5e9 * param
GPT2_LAYERS = 48
GPT2_HIDDEN_DIM = 1600
GPT2_HEADS = 25

# GPT-3 (175B)
GPT3_PARAMS = 175e9 * param
GPT3_LAYERS = 96
GPT3_HIDDEN_DIM = 12288
GPT3_HEADS = 96
GPT3_TRAINING_OPS = 3.14e23 * flop
GPT3_TRAINING_TOKENS = 300e9 * count
GPT3_TRAINING_DAYS_REF = 25 * day # Days on 1024 A100s
GPT3_TRAINING_ENERGY_MWH = 1287 # MWh, estimated per Patterson et al. (2021)

# GPT-4 (Reference) - Note: Unofficial public estimates
GPT4_EST_PARAMS = 1.76e12 * param
GPT4_LAYERS = 120
GPT4_HIDDEN_DIM = 16384
GPT4_HEADS = 128
GPT4_TRAINING_GPU_DAYS = 2.5e6 # A100 days

# Llama-2 (70B) — Source: Touvron et al. (2023)
LLAMA2_70B_PARAMS = 70e9 * param
LLAMA2_70B_LAYERS = 80
LLAMA2_70B_HIDDEN_DIM = 8192
LLAMA2_70B_HEADS = 64

# Llama 3.1
LLAMA3_8B_PARAMS = 8.03e9 * param
LLAMA3_8B_LAYERS = 32
LLAMA3_8B_HIDDEN_DIM = 4096
LLAMA3_8B_HEADS = 32
LLAMA3_8B_KV_HEADS = 8
LLAMA3_70B_PARAMS = 70.6e9 * param
LLAMA3_70B_LAYERS = 80
LLAMA3_70B_HIDDEN_DIM = 8192
LLAMA3_70B_HEADS = 64
LLAMA3_70B_KV_HEADS = 8
LLAMA3_405B_PARAMS = 405e9 * param

# BERT-Base — Source: Devlin et al. (2018)
BERT_BASE_PARAMS = 110e6 * param
BERT_BASE_LAYERS = 12
BERT_BASE_HIDDEN_DIM = 768
BERT_BASE_HEADS = 12
BERT_BASE_FLOPs = 22e9 * flop              # Per inference (seq_len=512)

# BERT-Large — Source: Devlin et al. (2018)
BERT_LARGE_PARAMS = 340e6 * param
BERT_LARGE_LAYERS = 24
BERT_LARGE_HIDDEN_DIM = 1024
BERT_LARGE_HEADS = 16
BERT_LARGE_FLOPs = 72e9 * flop             # Per inference (seq_len=512)

# AlexNet (Reference)
ALEXNET_PARAMS = 60e6 * param
ALEXNET_FLOPs = 1.5e9 * flop               # Estimated per inference

# YOLOv8-nano — Source: Ultralytics (2023)
YOLOV8_NANO_PARAMS = 3.2e6 * param
YOLOV8_NANO_FLOPs = 8.7e9 * flop  # 640x640
YOLOV8_NANO_LAYERS = 225

# WakeVision (Doorbell) — Source: Banbury et al. (2021)
WAKEVISION_PARAMS = 0.25e6 * param
WAKEVISION_FLOPs = 25e6 * flop

# Mamba-130M — Source: Gu & Dao (2023)
MAMBA_130M_PARAMS = 130e6 * param
MAMBA_130M_LAYERS = 24
MAMBA_130M_HIDDEN_DIM = 768
MAMBA_130M_STATE_SIZE = 16

# Mamba-2.8B — Source: Gu & Dao (2023)
MAMBA_2_8B_PARAMS = 2.8e9 * param
MAMBA_2_8B_LAYERS = 64
MAMBA_2_8B_HIDDEN_DIM = 2560
MAMBA_2_8B_STATE_SIZE = 16

# Stable Diffusion v1.5 — Source: Rombach et al. (2022)
STABLE_DIFFUSION_V1_5_PARAMS = 860e6 * param
STABLE_DIFFUSION_V1_5_RESOLUTION = 512
STABLE_DIFFUSION_V1_5_STEPS = 50
STABLE_DIFFUSION_V1_5_FLOPs_PER_STEP = 20e9 * flop

# Reference model/dataset dimensions
TRANSFORMER_HIDDEN_DIM_EXAMPLE = 768
TRANSFORMER_SEQ_LEN_EXAMPLE = 512
TRANSFORMER_HEADS_EXAMPLE = 12
SYSTOLIC_ARRAY_DIM = 128
SIMD_REGISTER_BITS = 512
FP32_BITS = 32
INT8_BITS = 8
MNIST_IMAGE_WIDTH = 28
MNIST_IMAGE_HEIGHT = 28

# Synthetic Data Constraints
SYNTHETIC_PROVENANCE_OVERHEAD = 0.4
SYNTHETIC_VERIFICATION_PASSES = 3

# Inference Scaling
LOGIC_WALL_REASONING_STEPS_EXAMPLE = 128

# Statistics
KS_TEST_COEFFICIENT = 1.36

# --- Deployment Tiers (Reference Envelopes) ---
CLOUD_LATENCY_RANGE_MS = "100-500"
EDGE_LATENCY_RANGE_MS = "10-100"
MOBILE_LATENCY_RANGE_MS = "5-50"
TINY_LATENCY_RANGE_MS = "1-10"

MOBILE_RAM_RANGE_GB = "8-16"
MOBILE_STORAGE_RANGE = "128 GB-1 TB"
MOBILE_TDP_RANGE_W = "3-5"

# Deployment tiers (reference capacities)
SMARTPHONE_RAM_GB = 8 * GB
MCU_RAM_KIB = 512 * KiB
CLOUD_MEM_GIB = 100 * GiB
MOBILE_MEM_GIB = 8 * GiB
TINY_MEM_KIB = 512 * KiB

# Communication assumptions
ALLREDUCE_FACTOR = 2
GPUS_PER_HOST = 8

# Google Search (Reference)
GOOGLE_SEARCHES_PER_DAY = 8.5e9
GMAIL_EMAILS_PER_DAY = 121e9

# ResNet-50
RESNET50_PARAMS = 25.6e6 * param
RESNET50_FLOPs = 4.1e9 * flop

# MobileNetV2
MOBILENETV2_PARAMS = 3.5e6 * param
MOBILENETV2_FLOPs = 0.3e9 * flop

# MobileNetV1
MOBILENET_V1_PARAMS = 4.2e6 * param

# KWS DS-CNN (Keyword Spotting Depthwise Separable CNN)
KWS_DSCNN_PARAMS = 200e3 * param
KWS_DSCNN_FLOPs = 20e6 * flop

# DLRM (Deep Learning Recommendation Model) — Meta benchmark
DLRM_EMBEDDING_ENTRIES = 25e9  # 25 Billion entries (dimensionless count)
DLRM_EMBEDDING_DIM = 128
DLRM_MODEL_SIZE_FP32 = 100 * GB  # Approximate total model size

# --- Hardware Unit Costs (Approximate, 2024 baseline) ---
V100_UNIT_COST = 10000 * USD
A100_UNIT_COST = 15000 * USD
H100_UNIT_COST = 30000 * USD
H200_UNIT_COST = 35000 * USD
B200_UNIT_COST = 40000 * USD
MI300X_UNIT_COST = 15000 * USD
T4_UNIT_COST = 2500 * USD
CEREBRAS_CS3_UNIT_COST = 2000000 * USD      # Approx. system cost

# --- Hardware TDP (where not already defined above) ---
TPUV5P_TDP = 300 * watt

# --- Storage (I/O Bandwidth) ---
NVME_SEQUENTIAL_BW = 7.0 * GB / second    # NVMe SSD sequential read (Gen 4)
SYSTEM_MEMORY_BW = 50 * GB / second        # DDR4/DDR5 typical

# --- Case Studies ---
WAYMO_DATA_PER_HOUR_LOW = 1 * TB / hour
WAYMO_DATA_PER_HOUR_HIGH = 19 * TB / hour

# --- Anomaly Detection Case Study ---
ANOMALY_MODEL_PARAMS = 270e3 * param
ANOMALY_MODEL_LATENCY = 10.4 * ureg.ms
ANOMALY_MODEL_AUC = 0.86
ANOMALY_MODEL_ENERGY = 516 * ureg.microjoule

# --- Additional Constants for ML Systems Chapter ---
BATTERY_CAPACITY_MAH = 3000 * ureg.milliampere_hour
BATTERY_VOLTAGE_V = 3.7 * ureg.volt
BATTERY_ENERGY_J = (BATTERY_CAPACITY_MAH * BATTERY_VOLTAGE_V).to(joule)

# TinyML Hardware (ESP32-CAM)
ESP32_RAM = 520 * KiB
ESP32_FLASH = 4 * MB
ESP32_POWER_MIN = 0.05 * watt
ESP32_POWER_MAX = 1.2 * watt
ESP32_PRICE = 10 * USD

# Edge Hardware (NVIDIA DGX/Workstation)
DGX_RAM = 128 * GB
DGX_STORAGE = 4 * TB
DGX_POWER = 200 * watt
DGX_PRICE_MIN = 3000 * USD
DGX_PRICE_MAX = 5000 * USD

# Cloud Hardware (TPU Pod)
TPU_POD_CHIPS = 4096
TPU_POD_MEM = 131 * TB
TPU_POD_POWER = 3 * ureg.megawatt

# --- Shared Precision Map ---
# Used by Engine, ServingModel, SynthesisSolver to map precision strings to byte widths.
PRECISION_MAP = {
    "fp32": BYTES_FP32,
    "fp16": BYTES_FP16,
    "int8": BYTES_INT8,
    "int4": BYTES_INT4,
}

# Fleet-Scale Constants (Volume II)
# Re-exported from defaults.py for backward compatibility.
from .defaults import *  # noqa: E402,F401,F403
