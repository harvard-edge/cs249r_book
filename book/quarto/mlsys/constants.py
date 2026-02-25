# constants.py
# The "Physics Engine" of Machine Learning Systems
# This file defines the single source of truth for hardware specifications,
# constants, and conversion factors used throughout the textbook.

import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# --- Dimensionless Scalars (Helpers) ---
QUADRILLION = 1e15
TRILLION = 1e12
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3
HUNDRED = 100

# --- Units ---
byte = ureg.byte
second = ureg.second
joule = ureg.joule
watt = ureg.watt
milliwatt = ureg.milliwatt
meter = ureg.meter
hour = ureg.hour
day = ureg.day
count = ureg.count

# Register data-scale aliases so .to(TB), .to(GB/second), etc. work
ureg.define('KB = 1e3 * byte')
ureg.define('MB = 1e6 * byte')
ureg.define('GB = 1e9 * byte')
ureg.define('TB = 1e12 * byte')
ureg.define('PB = 1e15 * byte')

KB = ureg.KB
MB = ureg.MB
GB = ureg.GB
TB = ureg.TB
PB = ureg.PB

# Common precision sizes
BYTES_FP32 = 4 * byte
BYTES_INT32 = 4 * byte
BYTES_FP16 = 2 * byte
BYTES_INT8 = 1 * byte
BYTES_INT4 = 0.5 * byte
BYTES_ADAM_STATE = 8 * byte

# Binary units (pint has kibibyte etc. built-in, register short aliases)
ureg.define('KiB = 1024 * byte')
ureg.define('MiB = 1048576 * byte')
ureg.define('GiB = 1073741824 * byte')
ureg.define('TiB = 1099511627776 * byte')

KiB = ureg.KiB
MiB = ureg.MiB
GiB = ureg.GiB
TiB = ureg.TiB

# --- Time (registered so .to(MS) scales magnitudes correctly) ---
ureg.define('MS = 1e-3 * second')
ureg.define('US = 1e-6 * second')
ureg.define('NS = 1e-9 * second')

MS = ureg.MS
ms = ureg.ms          # pint built-in millisecond (alias for convenience)
US = ureg.US
NS = ureg.NS

# Common time conversions (unitless scalars)
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
SEC_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
HOURS_PER_DAY = 24
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
SEC_PER_YEAR = SEC_PER_DAY * DAYS_PER_YEAR
SEC_PER_YEAR_LEAP = int(365.25 * SEC_PER_DAY)
HOURS_PER_YEAR = 8760

# Data size scalars
BITS_PER_BYTE = 8
KIB_TO_BYTES = 1024
MIB_TO_BYTES = 1024 * 1024
GIB_TO_BYTES = 1024 * 1024 * 1024

# Time scalars
MS_PER_SEC = 1000

# --- Hardware Specifications (The Silicon Contract) ---

# FLOPs are dimensionless "operations"
ureg.define('flop = 1 * count')
ureg.define('KFLOPs = 1e3 * flop')
ureg.define('MFLOPs = 1e6 * flop')
ureg.define('GFLOP = 1e9 * flop')
ureg.define('GFLOPs = 1e9 * flop')
ureg.define('TFLOP = 1e12 * flop')
ureg.define('TFLOPs = 1e12 * flop')
ureg.define('PFLOPs = 1e15 * flop')
ureg.define('ZFLOPs = 1e21 * flop')

flop = ureg.flop
KFLOPs = ureg.KFLOPs
MFLOPs = ureg.MFLOPs
GFLOP = ureg.GFLOP
GFLOPs = ureg.GFLOPs
TFLOP = ureg.TFLOP
TFLOPs = ureg.TFLOPs
PFLOPs = ureg.PFLOPs
ZFLOPs = ureg.ZFLOPs

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

# NVIDIA B100/B200 (Blackwell, 2024) — Source: NVIDIA Blackwell Architecture
B200_FLOPS_FP16_TENSOR = 4500 * TFLOPs / second
B200_FLOPS_FP8_TENSOR = 9000 * TFLOPs / second
B200_MEM_BW = 8 * TB / second             # HBM3e
B200_MEM_CAPACITY = 192 * GiB
B200_TDP = 1000 * watt

# NVIDIA T4 (Turing, 2018) — Source: NVIDIA T4 Data Sheet
T4_FLOPS_FP16_TENSOR = 65 * TFLOPs / second
T4_FLOPS_INT8 = 130 * TFLOPs / second
T4_MEM_BW = 320 * GB / second
T4_TDP = 70 * watt

# Google TPU v4 — Source: Google TPUv4 paper (Jouppi et al., 2023)
TPUV4_FLOPS_BF16 = 275 * TFLOPs / second
TPUV4_MEM_BW = 1200 * GB / second

# Google TPU v5p — Source: Google Cloud Documentation (2024)
TPUV5P_FLOPS_BF16 = 459 * TFLOPs / second
TPUV5P_MEM_BW = 2.76 * TB / second
TPUV5P_MEM_CAPACITY = 95 * GiB
TPUV5P_ICI_BW = 1600 * GB / second        # Inter-Chip Interconnect

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
MOBILE_NPU_TOPS_INT8 = 35 * TFLOPs / second
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
ureg.define('Gbps = 1e9 * bit / second')
Gbps = ureg.Gbps
NETWORK_10G_BW = 10 * Gbps
NETWORK_100G_BW = 100 * Gbps
NETWORK_5G_ENERGY_PER_MB_MJ = 100 * ureg.millijoule / MB

# Intra-node interconnects
NVLINK_V100_BW = 300 * GB / second        # NVLink 2.0 (V100, 6 links × 50 GB/s)
NVLINK_A100_BW = 600 * GB / second        # NVLink 3.0 (A100, 12 links × 50 GB/s)
NVLINK_H100_BW = 900 * GB / second        # NVLink 4.0 (H100, 18 links × 50 GB/s)
PCIE_GEN3_BW = 15.75 * GB / second        # PCIe Gen3 x16 (after 128b/130b encoding)
PCIE_GEN4_BW = 32 * GB / second           # PCIe Gen4 x16 (bidirectional)
PCIE_GEN5_BW = 64 * GB / second           # PCIe Gen5 x16 (bidirectional)

# Inter-node interconnects
INFINIBAND_HDR_BW = 200 * Gbps            # HDR InfiniBand (25 GB/s)
INFINIBAND_NDR_BW = 400 * Gbps            # NDR InfiniBand (50 GB/s)

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

# --- Physics ---
SPEED_OF_LIGHT_FIBER_KM_S = 200000 * ureg.kilometer / second

# --- Cloud Pricing ---
ureg.define('dollar = 1 * count')
USD = ureg.dollar
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
ureg.define('param = 1 * count')
ureg.define('Kparam = 1e3 * param')
ureg.define('Mparam = 1e6 * param')
ureg.define('Bparam = 1e9 * param')

param = ureg.param
Kparam = ureg.Kparam
Mparam = ureg.Mparam
Bparam = ureg.Bparam

# GPT-2 (1.5B) — used in training chapter worked examples
GPT2_PARAMS = 1.5e9 * param
GPT2_LAYERS = 48
GPT2_HIDDEN_DIM = 1600

# GPT-3 (175B)
GPT3_PARAMS = 175e9 * param
GPT3_TRAINING_OPS = 3.14e23 * flop
GPT3_TRAINING_TOKENS = 300e9 * count
GPT3_TRAINING_DAYS_REF = 25 # Days on 1024 A100s

# GPT-4 (Reference)
GPT4_TRAINING_GPU_DAYS = 2.5e6 # A100 days

# BERT-Base
BERT_BASE_PARAMS = 110e6 * param
BERT_BASE_FLOPs = 22e9 * flop              # Per inference (seq_len=512)
BERT_LARGE_PARAMS = 340e6 * param

# AlexNet (Reference)
ALEXNET_PARAMS = 60e6 * param

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

# Statistics
KS_TEST_COEFFICIENT = 1.36

# Deployment tiers (reference capacities)
SMARTPHONE_RAM_GB = 6 * GB
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

# YOLOv8-nano
YOLOV8_NANO_FLOPs = 8.7e9 * flop  # 640x640

# --- Storage (I/O Bandwidth) ---
NVME_SEQUENTIAL_BW = 3.5 * GB / second    # NVMe SSD sequential read
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
ESP32_POWER_MAX = 0.25 * watt
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
